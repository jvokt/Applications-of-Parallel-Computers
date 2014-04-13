##
# % Julia topic modeling via anchor words
# % David Bindel <bindel@cornell.edu>
# % 2014-04-06
#
# Introduction
# ============
# 
# This code computes a word-topic matrix using the spectral approach
# described in an [ICML 2013][1] paper.  The main algorithm has two
# phases:
#
# 1. In the first phase, we compute "anchor words" associated
#    with specific topics.
#
# 2. In the second phase, we compute the word-topic intensities
#    by a sequence of constrained QPs
#
# [1]: http://mimno.infosci.cornell.edu/papers/arora13.pdf
#
#
# Anchor word algorithms
# ======================
#
# Consider the non-negative factorization of a matrix $A$ as
# $A \approx W R$, where $W$ and $R$ are elementwise non-negative.
# This task becomes simple if $W$ can be permuted into the form
# $W = \begin{bmatrix} I \\ W_2 \end{bmatrix}$.  In this case,
# $R$ corresponds to a subset of the rows of the original matrix
# $A$, and these rows can be recovered by doing QR with column
# pivoting on $A^T$.
#
#
# Dense QRP
# ---------
#
# The `choose_anchors` routine uses a dense pivoted QR factorization
# to select anchor words.  The anchor words (permutation indices)
# and the diagonal of $R$ are returned as outputs.
#
# Julia provides a natural interface to the pivoted QR factorization
# with the `qrpfact` routine.  Note that we never need to form the
# orthogonal factor explicitly; we only care about the permutation
# (and to a lesser extent the triangular factor).  For moderate
# size problems, this will probably be the fastest option, since it
# is capable of taking advantage of level 3 BLAS operations; see 
# [Quintana-Orti, Sun, and Bischof][2].
#
# [2]: http://www.netlib.org/lapack/lawnspdf/lawn114.pdf

function choose_anchors_qrp(A)
  F = qrpfact(A)
  p = F[:p]
  r = abs(diag(F[:R]))
  p,r
end

##
# Partial QRP
# -----------
#
# If we only care about $k$ topics, where $k$ is significantly less than
# the number of words, then a full pivoted QR factorization is overkill.
# The `choose_anchors_partial` instead does a partial QR with deferred
# updating; this mostly involves keeping track of the column norms in
# an implicit fashion.  Note that this algorithm works well with large
# vocabularies (large, sparse A) as long as the number of topics remains
# modest.
#
# This is not the most numerically stable implementation possible.  A
# less quick-and-dirty version would do the orthogonalization more
# carefully (using at least MGS and perhaps Householder transforms for
# the orthogonalization step).  Nonetheless, it's fine for the current
# purpose.

function choose_anchors_partial(A, k)
  cnorms2 = sum(A.^2, 1)
  (m,n) = size(A)
  Q = zeros(Float64, (m,k))
  p = zeros(Integer, k)
  r = zeros(Float64, k)
  for j = 1:k
    p[j] = indmax(cnorms2)
    Q[:,j] = A[:,p[j]] - Q[:,1:j-1]*(Q[:,1:j-1]'*A[:,p[j]])
    r[j] = norm(Q[:,j])
    Q[:,j] = Q[:,j]/r[j]
    cnorms2 = cnorms2 - (Q[:,j]'*A).^2
  end
  p, r
end

##
# Computes the gradient for ||Tx-b||
function computeGradient(T, x, b)
    # Compute and return the gradient as defined in writeup
    p = 2 * transpose(T) * (T * x - b)
    return p
end

##
# Simplex-constrained QPs
# =======================
# 
# After we find the anchor words, the remaining expensive computation
# involves solving linearly constrained quadratic programs of the form
# $$
#   \min_x \{ \| \tilde{A}x-b \|^2 : x_i \geq 0, \sum_i x_i = 1 \}
# $$
# where $b^T$ is a row of $A$ and $\tilde{A}^T$ is a subset of the
# rows of $A$ corresponding to the anchor words.  The constraints
# force the vector $x$ to be interpretable as a probability distribution
# over the topics/anchor words.
#
# We provide two different algorithms for solving this quadratic program:
# exponentiated gradients and a primal active-set algorithm.
#
# Exponentiated gradients
# -----------------------
# 
# The exponentiated gradient algorithm was popularized in a [1996
# paper of Kivinen and Warmuth][3], where it was proposed as a good
# choice for online learning problems.  The basic idea is to replace the
# additive update of gradient descent with a multiplicative update.
# The algorithm is straightforward to implement, and it requires no
# work to incorporate the non-negativity constraints.  However,
# convergence is modest, and it is not entirely obvious how to choose
# the learning rate $\eta$.
#
# The `simplex_nnls_eg` routine solves the constrained quadratic
# program using the exponentiated gradient approach.  The routine
# effectively works with the normal equations, so we $A^T A$ and $A^T b$
# as algorithms.  A starting value may be provided as a third argument;
# the default starting point is a uniform distribution.
#
# [3]: http://dx.doi.org/10.1006/inco.1996.2612

function simplex_nnls_eg(AtA,Atb,x=[])
    # BEGIN TASK

    # Parameters for algorithm
    eta = 1 # Learning rate
    epsilon = 1 # Convergence terminator
    maxTime = 1000 * 60 * 2 # Max time to run algo (in ms, 2 min)

    # Compute the K value which is the num columns of AtA
    K = size(AtA,2)

    # If x has not been initialized (incorrect size), start with uniform
    if size(x,1) != K
        # Incorrect size, reset to uniform
        x = Array(Float64, K)
        fill!(x, 1.0/K)
    end

    # Compute an initial gradient
    p = computeGradient(AtA, x, Atb)

    # Run until the algorithm converges
    isConverged = false
    numIterations = 0
    endTime = time() + maxTime
    while !isConverged && time() < endTime

        # Perform a component wise multiplicative update
        for k = 1:K
            x[k] = x[k] * e^(-1 * eta * p[k])
        end

        # Project onto simplex
        x = x / norm(x,1)

        # Compute new gradient
        pPrime = computeGradient(AtA, x, Atb)

        # Test convergence based on last iteration (or initial) gradient
        minComponentDiff = pPrime - (minimum(a) * ones(Float64, size(pPrime)))
        convergeAmount = transpose(minComponentDiff) * x
        if convergeAmount < epsilon
            # Algorithm has converged
            isConverged = true
        end

        # Increase the iteration count & keep track of gradient
        numIterations += 1
        p = pPrime
    end

    # Return the final simplex ci
    return (x, numIterations)

    # END TASK
end

##
# Active set algorithm
# --------------------
# 
# Exponentiated gradients are fast for modest accuracy and scale to
# large problems.  However, the learning rate is not so obvious
# (at least to me!), and the convergence is modest.  In contrast,
# primal active-set algorithms provide high accuracy solutions,
# but quickly become too expensive for large problems.  It's still
# worth having a high-accuracy method around for testing, though,
# and it is possible to make this approach much faster by treating
# the linear solves involved in the direction computation more
# intelligently (or by using a quasi-Newton step instead of a Newton
# step).

function simplex_nnls_as(AtA, Atb, x=[])

  # Set up starting point and tolerance
  if isempty(x)
    x = ones( eltype(Atb), size(Atb) )
  end
  x = max(x, 0)
  x = x / sum(x)
  tol = 1e-12
  maxiter = 1000

  p = zeros(size(x))
  alphas = zeros(size(x))
  is_active = (x .<= 0)
  R = find(is_active)
  P = find(!is_active)

  # Main loop
  for i = 1:maxiter

    # Solve the equality constrained problem for a search dir
    r = AtA*x-Atb
    eP = ones(size(P))
    pm = [ AtA[P,P] eP ; eP' 0 ] \ [ r[P] ; 0 ]
    mu = pm[end]
    p[P] = pm[1:end-1]
    p[R] = 0

    if norm(p[P], Inf) < tol

      # Compute multipliers on inequality constraints
      lambda = mu.-r[R]

      # Release a constraint that should not be active
      if any(lambda .> -tol)
        (lmax,m) = findmax(lambda)
        m = R[m]
        is_active[m] = false
        R = find(is_active)
        P = find(!is_active)
      else
        return x
      end

    else

      # Take a Newton step
      I = find(p .> 0)
      if ~isempty(I)
        (alpha,m) = findmin(x[I]./p[I])
        m = I[m]
        if alpha >= 1
          x = x - p
        else
          x = x - alpha*p
          is_active[m] = true
          R = find(is_active)
          P = find(!is_active)
        end
      else
        x = x - p
      end
      x = max(x, 0)
      x = x/sum(x)

    end

  end  
  println("Did not converge")
  x
end

##
# Simplex projection
# ------------------
#
# The exponentiated gradient algorithm and the primal active
# set both have the property that they never consider violating the
# inequality constraints.  An alternative is a *projected gradient*
# or related method.  While we might not do projected gradients here,
# it's at least something to consider.  The only tricky part is
# efficient computation of the projection of some point onto
# the simplex; we provide a `proj_simplex` algorithm based on
# a [paper of Chen and Ye][4] that accomplishes this.
#
# Simplex projection can also be useful for choosing a starting point
# for the optimization.
#
# [4]: http://arxiv.org/pdf/1101.6081v2.pdf

function proj_simplex(y)
  (n,) = size(y)
  ys = sort(y,1)
  zs = flipud(cumsum(flipud(ys)))
  for i = n-1:-1:1
    t = (zs[i+1]-1)/(n-i)
    if t >= ys[i]
      return max(y-t,0)
    end
  end
  t = (zs[1]-1)/n
  max(y-t,0)
end

##
# Compute intensities
# ===================
#
# The row-normalized matrix $\bar{Q}$ can be interpreted as
# $$ 
#   \bar{Q}_{ij} = 
#     P(\mbox{word}_1 = i | \mbox{word}_2 = j).
# $$
# We suppose that any row of $\bar{Q}$ can be well approximated
# as a convex combination of anchor words $\bar{Q}(p,:)$ with
# weights summing to one, i.e.
# $$
#   \bar{Q}(i,:) \approx C(i,:) \bar{Q}(p,:)
# $$
# where the coefficients $C$ belong to the simplex.  We can find the
# optimal $C(i,:)$ for each row $i$ (in a least squares sense) by
# solving constrained quadratic programs using one of the
# `simplex_nnls` routines.
#
# We interpret each coefficient $C_{ik}$ as the conditional
# probability of the topic $k$ given word $i$, but what we want
# is the conditional probability of word $i$ given topic $k$.
# To compute this by Bayes rule, we need the overall probability
# of word $i$, which we store as $s$.

function compute_A(Qn, s, p)
  Tt = Qn[p,:]
  AtA = full(Tt*Tt')
  AtB = full(Tt*(Qn'))
  (nt,nw) = size(Tt)
  C = zeros(Float64, (nw,nt))
  maxerr1 = 0.0
  maxerr2 = 0.0
  for i = 1:nw
    Atb = reshape(full(AtB[:,i]), (nt,))

    # Version 1: Exponentiated gradient
    #(ci, maxiter) = simplex_nnls_eg(AtA,Atb)

    # Version 2: Warm-started active-set iteration
    ci = proj_simplex(AtA\Atb)
    ci = simplex_nnls_as(AtA, Atb, ci)
    C[i,:] = ci' .* s[i]

    # Check normalization error
    maxerr1 = max(maxerr1, abs(sum(ci)-1))
    
    # Check error measure used in EG convergence
    r = AtA*ci-Atb
    phi = 2*(r.-minimum(r))'*ci
    maxerr2 = max(maxerr2, phi[1])

  end
  println("Max error ", maxerr1, " ", maxerr2)
  sc = reshape(sum(C,1),nt)
  scale(C,1./sc)
end

##
# The main event
# ==============
#
# The overall topic mining algorithm is:
# 
# 0. Row normalize Q and compute word probabilities (row sums)
# 1. Select anchor words for each topic
# 2. Compute an intensity matrix associated with those words
# 3. Find the most important words for each topic
#
# As input, we need the fully normalized word co-occurrence
# matrix $Q$.  We output the anchor word selection $p$ (and
# "score" vector $r$), the intensity matrix $A$, and the
# top word selection `TW`.

function mine_topics(Q, ntopic=100, nword=20)

  println("-- Compute row scaling")
  tic()
  s = sum(Q,2)
  s = s[:,1]
  Qn = scale(1./s, Q)
  toc()

  println("-- Timing anchor words with partial fact")
  tic(); (p,r) = choose_anchors_partial(Qn', ntopic); toc()

  println("-- Compute intensities")
  tic(); A = compute_A(Qn, s, p); toc()

  println("-- Find top words per topic")
  tic()
  TW = zeros(Integer, (nword,ntopic))
  for t = 1:ntopic
    tp = sortperm(-A[:,t])
    TW[:,t] = tp[1:nword]
  end
  toc()
  p, r, A, TW

end

##
# Preprocessing code
# ==================
# 
# The raw data from the UCI sets (or bag-of-words in general) is a
# word-document frequency count matrix $D$.  We want to compute from
# this the word co-occurrence matrix; we may also want to trim
# unimportant (low frequency) words from the vocabulary.
#
# Basic word-document statistics
# ------------------------------

function doc_lengths(docs, Ddw)
  counts = zeros(eltype(docs), (maximum(docs),))
  for k = 1:prod(size(docs))
    counts[docs[k]] = counts[docs[k]] + Ddw[k]
  end
  counts
end

function word_freqs(words, Ddw)
  counts = zeros(eltype(words), (maximum(words),))
  for k = 1:prod(size(words))
    counts[words[k]] = counts[words[k]] + Ddw[k]
  end
  counts
end

function word_idfs(words, docs)
  counts = zeros(eltype(words), (maximum(words),))
  for k = 1:prod(size(words))
    counts[words[k]] = counts[words[k]] + 1
  end
  log(maximum(docs)) .- log(counts)
end

##
# Compacting the document matrix
# ------------------------------
# 
# If words or documents are pruned from a data set, the remaining
# term-document matrix may have empty rows or columns that we
# want to discard.

function compact_doc_matrix(vocab, docs, words, Ddw)

  # Re-index documents
  nd = doc_lengths(docs, words)
  active = find(nd .> 0)
  map_doc_id = zeros(eltype(docs), size(nd))
  map_doc_id[active] = 1:prod(size(active))
  docs = map_doc_id[docs]

  # Re-index terms
  tf = word_freqs(words, Ddw)
  active = find(tf .> 0)
  map_word_id = zeros(eltype(words), size(tf))
  map_word_id[active] = 1:prod(size(active))
  words = map_word_id[words]

  # Discard unused words from the vocabulary
  vocab = vocab[active]

  vocab, docs, words, Ddw
end

function trim_vocab(vocab, docs, words, Ddw, nwords)

  # Re-order the vocabulary by descending importance (tf-idf)
  tf  = word_freqs(words, Ddw)
  idf = word_idfs(words, docs)
  p = sortperm(tf .* idf, rev=true)
  pinv = zeros(eltype(p), size(p))
  pinv[p] = 1:prod(size(p))
  vocab = vocab[p]
  words = pinv[words]

  # Keep only entries for "important" words
  I = find(words .<= nwords)
  println("Trim: nwords=", prod(size(vocab)), " -> nwords=", nwords)
  println("Trim: nnz=", prod(size(Ddw)), " -> nnz=", prod(size(I)))
  docs, words, Ddw = docs[I], words[I], Ddw[I]

  # Compact the term-doc matrix and return
  compact_doc_matrix(vocab, docs, words, Ddw)
end

##
# Computing the word co-occurrence matrix
# ---------------------------------------
#
# The construction of $Q$ is based on the description in
# [the supplementary material from Arora 2013][5].
#
# [5]: http://cs.nyu.edu/~dsontag/papers/AroraEtAl_icml13_supp.pdf

function compute_Q(docs, words, Ddw)
  println("-- Forming Q")
  tic()
  nd = doc_lengths(docs, Ddw)

  # Compute the matrices and sum across documents to get Hhat
  scaling = 1./(nd .* (nd-1))
  Hdw = Ddw .* scaling[docs]
  diag_Hhat = zeros(Float64, (maximum(words),))
  for i = 1:prod(size(Hdw))
    diag_Hhat[words[i]] = diag_Hhat[words[i]] + Hdw[i]
  end

  # Compute the Htilde
  scaling = 1./sqrt(nd .* (nd-1))
  Htdw = Ddw .* scaling[docs]
  Ht = sparse(words, docs, Htdw)
  Q = Ht*Ht' - spdiagm(diag_Hhat,0)
  
  Q = Q/( sum(Q)[1] )
  toc()
  Q
end

##
# I/O code
# ========
#
# Loading from the UCI data
# -------------------------
# 
# The [several bag-of-words data sets][6] at UCI are a potentially
# useful testing ground for this code.  We provide a loader that loads
# the words (trivial) and produces the normalized word co-occurrence
# matrix in compressed sparse column form.  Notice that the docword
# file should be left gzipped (as it is on the web page).
# 
# [6]: https://archive.ics.uci.edu/ml/datasets/Bag+of+Words

using GZip

function load_uci(basename, nwords=0)
  vocab = "uci/vocab.$basename.txt"
  docword = "uci/docword.$basename.txt.gz"

  println("-- Reading files")
  tic()
  words = readcsv(vocab)
  f = gzopen(docword, "r")
  d = int(readline(f))
  w = int(readline(f))
  nnz = int(readline(f))
  idocs = zeros(Int64, (nnz,))
  iwords = zeros(Int64, (nnz,))
  V = zeros(Float64, (nnz,))
  for i = 1:nnz
    entries = map(int, split(readline(f), (' ')))
    idocs[i] = entries[1]
    iwords[i] = entries[2]
    V[i] = entries[3]
  end
  close(f)
  toc()

  if nwords > 0
    println("-- Trimming vocabulary")
    tic()
    (words, idocs, iwords, V) = trim_vocab(words, idocs, iwords, V, nwords)
    toc()
  end

  words, compute_Q(idocs, iwords, V)
end

##
# Writing topic summaries
# -----------------------

function write_topics(fname, words, p, r, A, TW)
  f = open(fname, "w")
  (nkeyword, ntopic) = size(TW)
  for t = 1:ntopic
    pt = p[t]
    @printf(f, "%s (%1.3e):\n", words[pt], r[t])
    for k = 1:nkeyword
      kw = TW[k,t]
      @printf(f, "\t%-15s (%1.3e)\n", words[kw], A[kw,t])
    end
    @printf(f, "\n")
  end
  close(f)
end

##
# Driver for UCI data sets
# ========================
#
# For convenience, we provide a driver for running the mining
# algorithm and writing topics for data sets from the UCI
# repository.

function driver_uci(basename, nwords=0, ntopic=100)
  (words,Q) = load_uci(basename, nwords)
  (p, r, A, TW) = mine_topics(Q, ntopic)
  write_topics("topics_$basename.txt", words, p, r, A, TW)
end
