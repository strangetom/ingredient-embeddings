#!/usr/bin/env/python3

import shlex
import subprocess
import tempfile


class VocabCount:
    bin = "bin/vocab_count"

    @classmethod
    def run(
        cls,
        corpus: str,
        *,
        verbose: int = 2,
        max_vocab: int | None = None,
        min_count: int = 0,
    ) -> str:
        """Run vocab_count tool to generate vocabulary and token count from corpus.

        Parameters
        ----------
        corpus : str
            Text file containing training corpus.
        verbose : int, optional
            Verbosity of tool output whilst running.
        max_vocab : int | None, optional
            Upper bound on vocabulary size, i.e. keep the <int> most frequent words.
            The minimum frequency words are randomly sampled so as to obtain an even
            distribution over the alphabet.
        min_count : int, optional
            Lower limit such that words which occur fewer than <int> times are
            discarded.

        Returns
        -------
        str
            Path of txt file vocab written to.
        """
        cmd = f"{cls.bin} -verbose {verbose} "

        if max_vocab:
            cmd += f"-max-vocab {max_vocab} "

        if min_count > 0:
            cmd += f"-min-count {min_count} "

        args = shlex.split(cmd)
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as out,
            open(corpus, "r") as inp,
        ):
            subprocess.run(args, stdout=out, stdin=inp)
            outputfile = out.name

        if verbose == 2:
            print(f"Vocab written to {outputfile}")
        return outputfile


class Cooccur:
    bin = "bin/cooccur"

    @classmethod
    def run(
        cls,
        corpus: str,
        *,
        verbose: int = 2,
        symmetric: int = 1,
        window_size: int = 15,
        vocab_file: str,
        memory: float = 4.0,
        distance_weighting: int = 1,
    ) -> str:
        """Run cooccur tool to calculate word-word cooccurrence statistics.

        Parameters
        ----------
        corpus : str
            Text file containing training corpus.
        verbose : int, optional
            Verbosity of tool output whilst running.
        symmetric : int, optional
            If <int> = 0, only use left context;
            if <int> = 1 (default), use left and right.
        window_size : int, optional
            Number of context words to the left (and to the right, if symmetric = 1);
            Default 15.
        vocab_file : str
            File containing vocabulary (truncated unigram counts, produced by
            'vocab_count').
        memory : float, optional
            Soft limit for memory consumption, in GB -- based on simple heuristic, so
            not extremely accurate; default 4.0
        distance_weighting : int, optional
            If <int> = 0, do not weight cooccurrence count by distance between words;
            if <int> = 1 (default), weight the cooccurrence count by inverse of distance
            between words.

        Returns
        -------
        str
            Path of bin file cooccurrences written to.
        """
        cmd = f"{cls.bin} -verbose {verbose} "
        cmd += f"-symmetric {symmetric} "
        cmd += f"-window-size {window_size} "
        cmd += f"-vocab-file {vocab_file} "
        cmd += f"-memory {memory} "
        cmd += f"-distance-weighting {distance_weighting}"
        args = shlex.split(cmd)
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".bin", delete=False) as out,
            open(corpus, "r") as inp,
        ):
            subprocess.run(args, stdout=out, stdin=inp)
            outputfile = out.name

        if verbose == 2:
            print(f"Cooccurrences written to {outputfile}")
        return outputfile


class Shuffle:
    bin = "bin/shuffle"

    @classmethod
    def run(
        cls,
        cooccur: str,
        *,
        verbose: int = 2,
        memory: float = 4.0,
        seed: int | None = None,
    ) -> str:
        """Run shuffle tool to shuffle entries of word-word cooccurrence files.

        Parameters
        ----------
        cooccur : str
            Word-word cooccurrence files.
        verbose : int, optional
            Verbosity of tool output whilst running.
        memory : float, optional
            Soft limit for memory consumption, in GB -- based on simple heuristic, so
            not extremely accurate; default 4.0
        seed : int | None, optional
            Random seed to use.  If not set, will be randomized using current time.

        Returns
        -------
        str
            Path of shuf.bin file shuffle cooccurrences written to.
        """
        cmd = f"{cls.bin} -verbose {verbose} "
        cmd += f"-memory {memory} "

        if seed is not None:
            cmd += f"-seed {seed} "

        args = shlex.split(cmd)
        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".shuf.bin", delete=False
            ) as out,
            open(cooccur, "r") as inp,
        ):
            subprocess.run(args, stdout=out, stdin=inp)
            outputfile = out.name

        if verbose == 2:
            print(f"Shuffled cooccurrence written to {outputfile}")
        return outputfile


class GloVe:
    bin = "bin/glove"

    @classmethod
    def run(
        cls,
        *,
        input_file: str,
        vocab_file: str,
        verbose: int = 2,
        write_header: int = 0,
        vector_size: int = 50,
        threads: int = 8,
        iter: int = 25,
        eta: float = 0.05,
        alpha: float = 0.75,
        x_max: float = 100.0,
        binary: int = 0,
        model: int = 2,
        save_file: str = "vectors",
        seed: int | None = None,
    ) -> str:
        """Run Glove tool to generate word embedding vectors.

        Parameters
        ----------
        input_file : str
            Shuffled cooccurrences file.
        vocab_file : str
            Vocabulary file.
        verbose : int, optional
            Verbosity of tool output whilst running.
        write_header : int, optional
            If 1, write vocab_size/vector_size as first line. Do nothing if 0 (default).
        vector_size : int, optional
            Dimension of word vector representations (excluding bias term); default 50.
        threads : int, optional
            Number of threads; default 8.
        iter : int, optional
            Number of training iterations; default 25.
        eta : float, optional
            Initial learning rate; default 0.05.
        alpha : float, optional
            Parameter in exponent of weighting function; default 0.75.
        x_max : float, optional
            Parameter specifying cutoff in weighting function; default 100.0.
        binary : int, optional
            Save output in binary format (0: text, 1: binary, 2: both); default 0.
        model : int, optional
            Model for word vector output (for text output only); default 2
            0: output all data, for both word and context word vectors,
               including bias terms.
            1: output word vectors, excluding bias terms.
            2: output word vectors + context word vectors, excluding bias terms.
            3: output word vectors and context word vectors, excluding bias terms;
               context word vectors are row-concatenated to the word vectors.
        save_file : str, optional
            Filename, excluding extension, for word vector output; default vectors.
        seed : int | None, optional
            Random seed to use.  If not set, will be randomized using current time.

        Returns
        -------
        str
            Path of file embeddings written to.
        """
        # Modify save file name to include dimensions and embedding type.
        save_file = f"{save_file}.{vector_size}d.glove"

        cmd = f"{cls.bin} -verbose {verbose} "
        cmd += f"-input-file {input_file} "
        cmd += f"-vocab-file {vocab_file} "
        cmd += f"-write-header {write_header} "
        cmd += f"-vector-size {vector_size} "
        cmd += f"-threads {threads} "
        cmd += f"-iter {iter} "
        cmd += f"-eta {eta} "
        cmd += f"-alpha {alpha} "
        cmd += f"-x-max {x_max} "
        cmd += f"-binary {binary} "
        cmd += f"-model {model} "
        cmd += f"-save-file {save_file} "

        if seed is not None:
            cmd += f"-seed {seed} "

        args = shlex.split(cmd)
        subprocess.run(args)

        if verbose == 2:
            print(f"Embeddings written to {save_file}")
        return save_file
