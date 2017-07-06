# tests.test_export
# Test the export module - to generate a corpus for machine learning.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Sun Feb 21 15:49:18 2016 -0500
#
# Copyright (C) 2016 Bengfort.com
# For license information, see LICENSE.txt
#
# ID: test_export.py [2988c53] benjamin@bengfort.com $

"""
Test the export module - to generate a corpus for machine learning.
"""

##########################################################################
## Imports
##########################################################################

import unittest
from unittest.mock import MagicMock
import shutil

from mongomock import MongoClient as MockMongoClient

from tempfile import mkdtemp, TemporaryDirectory

from baleen.export import *
from baleen.feed import *
from baleen.models import connect

BOOKS_FEED = Feed(
    title='The Rumpus.net',
    link='http://therumpus.net/feed/',
    urls={'htmlurl': 'http://therumpus.net'},
    category='books',
)
BOOKS_POST = Post(
    feed=BOOKS_FEED,
    title="My Awesome Post",
    content="books",
    url="http://example.com/books.html",
)

POLITICS_FEED = Feed(
    title='The Politics Site',
    link='http://politicsrock.net/feed/',
    urls={'htmlurl': 'http://politicsrock.net'},
    category='politics',
)
POLITICS_POST = Post(
    feed=POLITICS_FEED,
    title="My Awesome Political Post",
    content="political parties",
    url="http://example.com/politics.html",
)

FOOD_FEED = Feed(
    title='I love food',
    link='http://foodisthebest.com/atom',
    urls={'htmlurl': 'http://foodisthebest.com'},
    category='food',
)
FOOD_POST = Post(
    feed=FOOD_FEED,
    title="Hamburgers are Good",
    content="ground meat",
    url="http://example.com/mmmmm.html",
)

CATEGORIES_IN_DB = [
    BOOKS_FEED.category,
    POLITICS_FEED.category,
    FOOD_FEED.category,
]


##########################################################################
## Export Tests
##########################################################################

class ExportTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create the mongomock connection
        """
        cls.conn = connect(host='mongomock://localhost')
        assert isinstance(cls.conn, MockMongoClient)

        # Clear out the database
        for feed in Feed.objects():
            feed.delete()
        for post in Post.objects():
            post.delete()
        assert Feed.objects.count() == 0
        assert Post.objects.count() == 0

        # Set up test feeds
        FeedSync(BOOKS_FEED).sync()
        BOOKS_POST.save()
        FeedSync(FOOD_FEED).sync()
        FOOD_POST.save()
        FeedSync(POLITICS_FEED).sync()
        POLITICS_POST.save()

        assert Feed.objects.count() == 3
        assert Post.objects.count() == 3

        # make a temp directory
        cls.root_dir = TemporaryDirectory(prefix=u"baleen")
        cls.corpus_dir = os.path.join(cls.root_dir.name, "corpus")
        cls.readme_path = os.path.join(cls.root_dir.name, "readme")


    @classmethod
    def tearDownClass(cls):
        """
        Drop the mongomock connection and clean up our temporary dir
        """
        assert isinstance(cls.conn, MockMongoClient)
        cls.conn = None
        cls.root_dir.cleanup()

    def setUp(self):
        pass

    def tearDown(self):
        if os.path.isdir(self.corpus_dir):
            shutil.rmtree(self.corpus_dir, ignore_errors=True)
        if os.path.isfile(self.readme_path):
            os.remove(self.readme_path)

    def test_scheme_specification(self):
        """
        Assert that only known schemes are allowed
        """

        # Make sure good schemes don't error
        for scheme in SCHEMES:
            try:
                exporter = MongoExporter(root=self.corpus_dir, scheme=scheme)
            except ExportError:
                self.fail("Could not use expected scheme, {}".format(scheme))

        # Make sure bad schemes do error
        for scheme in ('bson', 'xml', 'yaml', 'foo', 'bar'):
            with self.assertRaises(ExportError):
                exporter = MongoExporter(root=self.corpus_dir, scheme=scheme)

    def test_categories_default(self):
        """
        Assert that categories are set to default when not provided
        """

        exporter = MongoExporter(root=self.corpus_dir)
        self.assertCountEqual(CATEGORIES_IN_DB, exporter.categories)

    def test_categories_provided(self):
        """
        Assert that provided categories are returned
        """
        categories = ["TestCategory", "Another Category", "Unicode ĆăƮĖƓƠŕƔ"]
        exporter = MongoExporter(root=self.corpus_dir, categories=categories)
        self.assertCountEqual(categories, exporter.categories)

    def test_feeds_for_list_of_categories(self):
        """
        Assert that getting feeds for a list of categories works
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)
        expected_feeds = [POLITICS_FEED, FOOD_FEED]
        test_categories = ["politics", "food"]
        self.assertCountEqual(expected_feeds, exporter.feeds(categories=test_categories))

    def test_feeds_for_category_string(self):
        """
        Assert that getting feeds for a category as a string
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)
        self.assertCountEqual([POLITICS_FEED], exporter.feeds(categories="politics"))

    def test_feeds_for_all_categories(self):
        """
        Assert that getting feeds with a category returns for all categories
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)
        self.assertCountEqual([POLITICS_FEED, FOOD_FEED, BOOKS_FEED], exporter.feeds())

    def test_writing_readme(self):
        """
        Assert that a readme file is written correctly
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)
        exporter.state = State.Finished
        exporter.readme(self.readme_path)
        self.assertTrue(os.path.isfile(self.readme_path))

    def test_writing_readme_fails(self):
        """
        Assert writing readme file fails when in an incorrect state
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)
        exporter.state = State.Started
        with self.assertRaises(ExportError):
            exporter.readme(self.readme_path)

    def test_generating_posts_fails(self):
        """
        Assert generating posts fails when in an incorrect state
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)
        exporter.state = "Some crazy thing"
        with self.assertRaises(ExportError):
            for post, category in exporter.posts():
                self.fail("Should never get to the point where we touch posts")

    """
    MockMongoClient doesn't have a rewind method:
    https://github.com/vmalloc/mongomock/blob/develop/mongomock/collection.py#L1498

    This forces us to mock the post() method here.
    """

    def test_export(self):
        """
        Assert that we can export posts
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)

        # Mock Mongo calls that aren't supported in MockMongoClient
        post_categories = [
            (BOOKS_POST, BOOKS_FEED.category),
            (FOOD_POST, FOOD_FEED.category),
            (POLITICS_POST, POLITICS_FEED.category),
        ]
        exporter.posts = MagicMock(return_value=post_categories)
        exporter.export()

    def test_export_with_root_path_failure(self):
        """
        Assert that root path failures are raised
        """
        root_path = self.corpus_dir
        exporter = MongoExporter(root_path, categories=CATEGORIES_IN_DB)
        os.path.exists = lambda path: False if path == root_path else True
        os.mkdir = lambda success: True  # Mock directory creation
        os.path.isdir = lambda path: False if path == root_path else True

        with self.assertRaises(ExportError):
            exporter.export()

    def test_export_with_category_path_failure(self):
        """
        Assert that category path failures are raised
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)

        for category in CATEGORIES_IN_DB:
            category_path = os.path.join(self.corpus_dir, category)
            os.path.exists = lambda path: False if path == category_path else True
            os.mkdir = lambda success: True  # Mock directory creation
            os.path.isdir = lambda path: False if path == category_path else True

        with self.assertRaises(ExportError):
            exporter.export()

    def test_export_with_invalid_sanitization(self):
        """
        Assert that export requires a valid level
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)

        with self.assertRaises(ExportError):
            exporter.export(level="BOGUS")

    def test_export_with_invalid_scheme(self):
        """
        Assert that export requires a valid scheme
        """
        exporter = MongoExporter(root=self.corpus_dir, categories=CATEGORIES_IN_DB)

        with self.assertRaises(ExportError):
            exporter.export(scheme="BOGUS")
