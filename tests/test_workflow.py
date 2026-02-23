#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted for academic and research use only (subject to the
#  limitations in the disclaimer below) provided that the following conditions are met:
#       * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#       * Neither the name of the copyright holders nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
#  THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
#  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
#  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import pytest

from clearex.workflow import WorkflowConfig, format_chunks, parse_chunks


class TestParseChunks:
    def test_parse_none(self):
        assert parse_chunks(None) is None

    def test_parse_empty(self):
        assert parse_chunks("") is None
        assert parse_chunks("   ") is None

    def test_parse_integer(self):
        assert parse_chunks("256") == 256

    def test_parse_tuple(self):
        assert parse_chunks("1,256,256") == (1, 256, 256)

    def test_reject_non_positive_values(self):
        with pytest.raises(ValueError):
            parse_chunks("0")
        with pytest.raises(ValueError):
            parse_chunks("1,0,64")

    def test_reject_non_numeric_values(self):
        with pytest.raises(ValueError):
            parse_chunks("abc")
        with pytest.raises(ValueError):
            parse_chunks("1,abc,64")


class TestFormatChunks:
    def test_format_none(self):
        assert format_chunks(None) == ""

    def test_format_integer(self):
        assert format_chunks(64) == "64"

    def test_format_tuple(self):
        assert format_chunks((1, 128, 128)) == "1,128,128"


class TestWorkflowConfig:
    def test_has_analysis_selection(self):
        cfg = WorkflowConfig()
        assert cfg.has_analysis_selection() is False

        cfg.registration = True
        assert cfg.has_analysis_selection() is True
