# baleen.export
# Export an HTML corpus for analyses with NLTK
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Fri Oct 03 16:49:20 2014 -0400
#
# Copyright (C) 2014 Bengfort.com
# For license information, see LICENSE.txt
#
# ID: export.py [eb962e7] benjamin@bengfort.com $

"""
Export an HTML corpus for analyses with NLTK
"""

##########################################################################
## Imports
##########################################################################

import codecs
import os
from collections import Counter
from datetime import datetime
from enum import Enum
from operator import itemgetter

from tqdm import tqdm

from baleen.exceptions import ExportError
from baleen.models import Feed, Post
from baleen.utils.text import sanitize_html, SAFE, SANITIZE_LEVELS

DTFMT = "%b %d, %Y at %H:%M"
JSON = 'json'
HTML = 'html'
SCHEMES = (JSON, HTML)
State = Enum('State', 'Init, Started, Finished')

INVALID_SANITIZE_LEVEL = "Unknown sanitization method: '{{}}' - use one of {LEVELS}.".format(
    LEVELS=(", ".join(SANITIZE_LEVELS)))
INVALID_EXPORT_SCHEME = "Unknown export format scheme '{{}}' - use one of {SCHEMES}.".format(
    SCHEMES=", ".join(SCHEMES))


##########################################################################
## Exporter
##########################################################################

class MongoExporter(object):
    """
    The exporter attempts to read the MongoDB as efficiently as possible,
    writing posts to disk in either HTML or JSON format.
    """

    def __init__(self, root, categories=None, scheme=JSON, sanitize_level=SAFE):
        scheme = scheme.lower()

        if not self.valid_scheme(scheme):
            raise ExportError(INVALID_EXPORT_SCHEME.format(scheme))
        else:
            self.scheme = scheme  # Output format of the data

        if not self.valid_sanitize_level(sanitize_level):
            raise ExportError(INVALID_SANITIZE_LEVEL.format(sanitize_level))
        else:
            self.sanitize_level = sanitize_level  # Sanitization to apply to content

        self.root = root  # Location on disk to write to
        self.state = State.Init  # Current state of the export
        self.counts = Counter()  # Counts of posts per category
        self.categories = categories  # Specific categories to export

    @property
    def categories(self):
        if self._categories is None:
            self._categories = Feed.objects.distinct('category')
        return self._categories

    @categories.setter
    def categories(self, value):
        self._categories = value

    def feeds(self, categories=None):
        """
        Returns a list of feeds for the specified categories.
        During export, this list is used to construct a feed-category mapping
        that is used to perform checking of sequential reads of Posts.
        """
        if isinstance(categories, str):
            categories = [categories]
        elif categories is None:
            categories = self.categories
        else:
            categories = list(categories)

        return Feed.objects(category__in=categories)

    def posts(self, categories=None):
        """
        This method first creates a mapping of feeds to categories, then
        iterates through the Posts collection, finding only posts with those
        given feeds (and not dereferencing the related object). This will
        speed up the post fetch process and give us more information, quickly.

        The generator therefore yields post, category tuples to provide for
        the single pass across the posts.

        This method also counts the number of posts per category.

        This method raises an exception if not in the correct state.
        """
        if self.state != State.Started:
            raise ExportError((
                "Calling the posts method when not in the started state "
                "could cause double counting or multiple database reads."
            ))

        # Create a mapping of feed id to category
        feeds = {
            feed.id: feed.category
            for feed in self.feeds(categories)
            }

        # Iterate through all posts that have the given feed ids without
        # dereferencing the related object. Yield (post, category) tuples.
        # This method also counts the number of posts per category.
        for post in Post.objects(feed__in=feeds.keys()).no_dereference().no_cache():
            category = feeds[post.feed.id]
            self.counts[category] += 1

            yield post, category

    def readme(self, path):
        """
        Writes README information about the state of the export to disk at
        the specified path. The writing requires the export to be finished,
        otherwise, the method will raise an exception.

        This method raises an exception if not in the correct state.
        """
        if self.state != State.Finished:
            raise ExportError((
                "Calling the readme method when not in the finished state "
                "could lead to writing misleading or incorrect meta data."
            ))

        # Create the output lines with the header information.
        output = [
            "Baleen RSS Export",
            "=================", "",
            "Exported on: {}".format(datetime.now().strftime(DTFMT)),
            "{} feeds containing {} posts in {} categories.".format(
                self.feeds().count(),
                sum(self.counts.values()),
                len(self.categories),
            ), "",
            "Category Counts",
            "---------------", "",
        ]

        # Append category counts list to the README
        for item in sorted(self.counts.items(), key=itemgetter(0)):
            output.append("- {}: {}".format(*item))

        # Add a newline at the end of the README
        output.append("")

        # Write out the output to the file as utf-8.
        with codecs.open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output))

    def feedinfo(self, path):
        """
        Writes information about the feeds to disk for performing lookups on
        the feeds themselves from the object id in each individual post.
        """
        fields = ('id', 'title', 'link', 'category', 'active')
        feeds = Feed.objects(category__in=self.categories).only(*fields)
        with open(path, 'w') as f:
            f.write(feeds.to_json(indent=2))

    def export(self, root=None, categories=None, scheme=JSON, level=SAFE):
        """
        Export all posts in categories to disk.
        """

        root = root or self.root
        categories = categories or self.categories

        scheme = scheme or self.scheme
        level = level or self.sanitize_level
        if not self.valid_scheme(scheme):
            raise ExportError(INVALID_EXPORT_SCHEME.format(scheme))

        if not self.valid_sanitize_level(level):
            raise ExportError(INVALID_SANITIZE_LEVEL.format(level))

        self.initialize_export_directory(root)

        category_filepaths = {}

        # Reset the counts object and mark export as started.
        self.counts = Counter()
        self.state = State.Started

        # Iterate through all posts, writing them to disk correctly.
        # Right now we will simply write them based on their object id.
        for post, category in tqdm(self.posts(categories=categories),
                                   total=Post.objects.count(),
                                   unit="docs"):
            if not hasattr(category_filepaths, category):
                category_filepaths[category] = self.initialize_category_dir(root, category)

            path = os.path.join(category_filepaths[category], "{}.{}".format(post.id, self.scheme))

            try:
                with codecs.open(path, 'w', encoding='utf-8') as f:
                    action = {
                        JSON: lambda: post.to_json(indent=2),
                        HTML: lambda: post.htmlize(sanitize=level)
                    }[self.scheme]

                    f.write(action())
            except FileNotFoundError as e:
                msg = "FileNotFoundError ({0}): {1}".format(e.errno, e.strerror)
                raise ExportError(msg)
        # Mark the export as finished and write the README to the corpus.
        self.state = State.Finished
        self.readme(os.path.join(root, "README"))
        self.feedinfo(os.path.join(root, "feeds.json"))

    @classmethod
    def initialize_category_dir(cls, base_dir, category):
        """
        Create a directory for a category
        :param base_dir: the absolute filepath to create the category directory in
        :param category: an iterable of category names
        :return: a string of the filepath
        """
        path = os.path.join(base_dir, category)
        cls.initialize_export_directory(path)
        return path

    @classmethod
    def initialize_export_directory(cls, directory):
        # Make the directory to export if it doesn't exist.
        if not os.path.exists(directory):
            os.mkdir(directory)

        # If the root is not a directory, then we can't write there.
        if not os.path.isdir(directory):
            raise ExportError(
                "'{}' is not a directory!".format(directory)
            )

    @classmethod
    def valid_sanitize_level(self, level):
        """
        :param level: sanitization level
        :return: Boolean
        """
        return (not level) or (level in SANITIZE_LEVELS)

    @classmethod
    def valid_scheme(cls, scheme):
        return (not scheme) or (scheme in SCHEMES)


if __name__ == '__main__':
    import baleen.models as db

    db.connect()
    exporter = MongoExporter('fixtures/corpus')
    exporter.export()
