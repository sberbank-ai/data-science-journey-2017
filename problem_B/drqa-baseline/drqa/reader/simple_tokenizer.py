#!/usr/bin/env python3
# Copyright 2017-present.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Basic tokenizer that splits text into alpha-numeric tokens and
non-whitespace tokens.
"""

import regex
import logging
import pymorphy2
from .tokenizer import Tokens, Tokenizer

logger = logging.getLogger(__name__)


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        annotators = kwargs.get('annotators', {})
        if len(annotators) > 0 and not ('lemma' in annotators and len(annotators) == 1):
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = annotators
        if 'lemma' in self.annotators:
            self.ma = pymorphy2.MorphAnalyzer()
        else:
            self.ma = None

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()
            if self.ma is not None:
                lemma = self.ma.parse(token)[0].normal_form
            else:
                lemma = None
            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
                None,
                lemma,
                None,
            ))
        return Tokens(data, self.annotators)
