# -*- coding: utf-8 -*-


class Tweet(object):
    def __init__(self):
        self._idTweet = None
        self._text = None
        self._procText = None
        self._tags = None
        self._procTags = None
        self._mentions = None
        self._geoCoord = None
        self._candidates = None
        self.entities = None

    @property
    def idTweet(self):
        return self._idTweet

    @idTweet.setter
    def idTweet(self, value):
        self._idTweet = value

    @idTweet.deleter
    def idTweet(self):
        del self._idTweet

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @text.deleter
    def text(self, value):
        del self._text

    @property
    def procText(self):
        return self._procText

    @procText.setter
    def procText(self, value):
        self._procText = value

    @procText.deleter
    def procText(self):
        del self._procText

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        self._tags = value

    @tags.deleter
    def tags(self):
        del self._tags

    @property
    def procTags(self):
        return self._procTags

    @tags.setter
    def procTags(self, value):
        self._procTags = value

    @tags.deleter
    def procTags(self):
        del self._procTags

    @property
    def mentions(self):
        return self._mentions

    @mentions.setter
    def mentions(self, value):
        self._mentions = value

    @mentions.deleter
    def mentions(self):
        del self._mentions

    @property
    def geoCoord(self):
        return self._geoCoord

    @geoCoord.setter
    def geoCoord(self, value):
        self._geoCoord = value

    @geoCoord.deleter
    def geoCoord(self):
        del self._geoCoord

    @property
    def candidates(self):
        return self._candidates

    @candidates.setter
    def candidates(self, value):
        self._candidates = value

    @candidates.deleter
    def candidates(self):
        del self._candidates

    @property
    def entities(self):
        return self._entities

    @entities.setter
    def entities(self, value):
        self._entities = value

    @entities.deleter
    def entities(self):
        del self._entities