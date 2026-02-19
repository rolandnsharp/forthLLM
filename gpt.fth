\ gpt.fth — Minimal GPT in Forth (requires gforth)
\ Translated from @karpathy's pure Python GPT
\ Run: gforth gpt.fth
\
\ "The most atomic way to train and run inference for a GPT in pure Forth."

\ ============================================================
\ Section 0: Configuration
\ ============================================================

1 constant N_LAYER
16 constant N_EMBD
16 constant BLOCK_SIZE
4 constant N_HEAD
N_EMBD N_HEAD / constant HEAD_DIM
0 value VOCAB_SIZE
0 value BOS_TOKEN
1000 constant NUM_STEPS
3.14159265358979323846e fconstant PI

\ ============================================================
\ Section 1: Memory Management
\ ============================================================

\ Temporary pool for computation graph nodes (reset each training step)
67108864 constant POOL_SIZE
variable pool-base
variable pool-ptr

: init-pool   POOL_SIZE allocate throw dup pool-base ! pool-ptr ! ;
: reset-pool  pool-base @ pool-ptr ! ;
: pool-alloc { n -- addr }
  \ Keep allocations 16-byte aligned to avoid float alignment faults.
  pool-ptr @ 15 + -16 and { a }
  a n + 15 + -16 and pool-ptr !
  a ;

\ ============================================================
\ Section 2: Random Number Generator
\ ============================================================

variable rng-state
42 rng-state !

: rng ( -- u )
  rng-state @ 6364136223846793005 * 1442695040888963407 +
  dup rng-state ! ;

: rng-float ( F: -- r )
  rng $FFFFFFFF and s>f 4294967296e f/ ;

: rng-gauss ( F: -- x )
  rng-float 1e-10 fmax fln -2e f* fsqrt
  rng-float PI 2e f* f* fcos f* ;

\ ============================================================
\ Section 3: Value Node (Autograd)
\ ============================================================

\ Layout: 64 bytes per node
\   +0  data     (float)    +8  grad     (float)
\  +16  n-children (cell)   +24 child0   (cell)
\  +32  child1   (cell)     +40 lgrad0   (float)
\  +48  lgrad1   (float)    +56 vis-gen  (cell)
64 constant VAL_SIZE

: v.data   ( v -- addr ) ;                \ offset 0
: v.grad   ( v -- addr ) 8 + ;
: v.nchild ( v -- addr ) 16 + ;
: v.child0 ( v -- addr ) 24 + ;
: v.child1 ( v -- addr ) 32 + ;
: v.lgrad0 ( v -- addr ) 40 + ;
: v.lgrad1 ( v -- addr ) 48 + ;
: v.vgen   ( v -- addr ) 56 + ;

\ Create a temporary Value (from pool, freed on reset-pool)
: new-val ( F: data -- ) ( -- v )
  VAL_SIZE pool-alloc { v }
  v v.data f!
  0e v v.grad f!
  0 v v.nchild !
  0 v v.vgen !
  v ;

\ Create a permanent Value (for parameters, persists across steps)
: new-pval ( F: data -- ) ( -- v )
  VAL_SIZE allocate throw { v }
  v v.data f!
  0e v v.grad f!
  0 v v.nchild !
  0 v v.vgen !
  v ;

\ --- Value operations ---

: val+ { a b -- r }
  a v.data f@ b v.data f@ f+ new-val { r }
  2 r v.nchild !
  a r v.child0 !   b r v.child1 !
  1e r v.lgrad0 f!  1e r v.lgrad1 f!
  r ;

: val* { a b -- r }
  a v.data f@ b v.data f@ f* new-val { r }
  2 r v.nchild !
  a r v.child0 !   b r v.child1 !
  b v.data f@ r v.lgrad0 f!
  a v.data f@ r v.lgrad1 f!
  r ;

: val** { a } ( F: n -- ) ( -- r )
  a v.data f@ { F: ad }  { F: n }
  ad n f** new-val { r }
  1 r v.nchild !
  a r v.child0 !
  n ad n 1e f- f** f* r v.lgrad0 f!
  r ;

: val-log { a -- r }
  a v.data f@ fln new-val { r }
  1 r v.nchild !
  a r v.child0 !
  1e a v.data f@ f/ r v.lgrad0 f!
  r ;

: val-exp { a -- r }
  a v.data f@ fexp { F: ev }
  ev new-val { r }
  1 r v.nchild !
  a r v.child0 !
  ev r v.lgrad0 f!
  r ;

: val-relu { a -- r }
  a v.data f@ { F: d }
  d f0> if d else 0e then new-val { r }
  1 r v.nchild !
  a r v.child0 !
  d f0> if 1e else 0e then r v.lgrad0 f!
  r ;

: val-neg { a -- r }
  a v.data f@ fnegate new-val { r }
  1 r v.nchild !
  a r v.child0 !
  -1e r v.lgrad0 f!
  r ;

: val- ( a b -- r )  val-neg val+ ;
: val/ { a b -- r }  a  b -1e val**  val* ;

\ ============================================================
\ Section 4: Backward Pass (iterative topological sort)
\ ============================================================

200000 constant MAX_GRAPH
create dfs-stk MAX_GRAPH 2 * cells allot   \ pairs: (node, phase)
variable dfs-sp
create topo-buf MAX_GRAPH cells allot
variable topo-n
variable topo-gen
0 topo-gen !

: backward { loss -- }
  1 topo-gen +!
  0 topo-n !
  0 dfs-sp !
  \ push (loss, 0)
  loss dfs-stk !  0 dfs-stk cell+ !  2 dfs-sp !
  begin dfs-sp @ 0> while
    \ pop (node, phase)
    -2 dfs-sp +!
    dfs-stk dfs-sp @ cells + @ { node }
    dfs-stk dfs-sp @ 1+ cells + @ { phase }
    phase 1 = if
      \ post-visit: add to topo list
      node topo-buf topo-n @ cells + !
      1 topo-n +!
    else
      node v.vgen @ topo-gen @ <> if
        topo-gen @ node v.vgen !
        \ push (node, 1) for post-visit
        node dfs-stk dfs-sp @ cells + !
        1    dfs-stk dfs-sp @ 1+ cells + !
        2 dfs-sp +!
        \ push children with phase 0
        node v.nchild @ { nc }
        nc 1 > if
          node v.child1 @ { c1 }
          c1 v.vgen @ topo-gen @ <> if
            c1 dfs-stk dfs-sp @ cells + !
            0  dfs-stk dfs-sp @ 1+ cells + !
            2 dfs-sp +!
          then
        then
        nc 0> if
          node v.child0 @ { c0 }
          c0 v.vgen @ topo-gen @ <> if
            c0 dfs-stk dfs-sp @ cells + !
            0  dfs-stk dfs-sp @ 1+ cells + !
            2 dfs-sp +!
          then
        then
      then
    then
  repeat
  \ Backpropagate: loss.grad = 1, then reverse topo order
  1e loss v.grad f!
  topo-n @ 0 ?do
    topo-n @ 1- i - { idx }
    topo-buf idx cells + @ { v }
    v v.nchild @ { nc }
    nc 0> if
      v v.child0 @ { c0 }
      c0 v.grad f@  v v.lgrad0 f@  v v.grad f@  f* f+  c0 v.grad f!
    then
    nc 1 > if
      v v.child1 @ { c1 }
      c1 v.grad f@  v v.lgrad1 f@  v v.grad f@  f* f+  c1 v.grad f!
    then
  loop ;

\ ============================================================
\ Section 5: Dataset Loading
\ ============================================================

32768 constant MAX_DOCS
20 constant MAX_NAME
create doc-buf MAX_DOCS MAX_NAME * chars allot
create doc-len MAX_DOCS cells allot
variable n-docs  0 n-docs !
create doc-idx MAX_DOCS cells allot  \ shuffled indices

: doc-addr ( i -- addr )  MAX_NAME * doc-buf + ;

: load-data
  s" input.txt" file-status nip 0<> if
    s" curl -sL 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt' -o input.txt"
    system
  then
  s" input.txt" r/o open-file throw { fid }
  0 n-docs !
  begin
    pad 256 fid read-line throw
  while
    n-docs @ MAX_DOCS < if
      dup 0> if
        { len }
        len MAX_NAME min { keep }
        pad n-docs @ doc-addr keep cmove
        keep doc-len n-docs @ cells + !
        1 n-docs +!
      else
        drop
      then
    else
      drop
    then
  repeat
  drop
  fid close-file drop
  ." num docs: " n-docs @ . cr ;

: init-indices
  n-docs @ 0 do  i doc-idx i cells + !  loop ;

: shuffle-indices
  n-docs @ { n }
  n 1 ?do
    n i - { k }
    rng $7FFFFFFFFFFFFFFF and k 1+ mod { j }
    \ swap doc-idx[k] and doc-idx[j]
    doc-idx k cells + @ { tmp }
    doc-idx j cells + @ doc-idx k cells + !
    tmp doc-idx j cells + !
  loop ;

\ ============================================================
\ Section 6: Tokenization
\ ============================================================

create uchars 128 chars allot   \ sorted unique characters
variable n-uchars  0 n-uchars !
create char-seen 256 chars allot

: build-vocab
  char-seen 256 erase
  n-docs @ 0 do
    doc-len i cells + @ { len }
    i doc-addr { addr }
    len 0 do
      1 char-seen addr i + c@ + c!
    loop
  loop
  0 n-uchars !
  256 0 do
    char-seen i + c@ if
      i uchars n-uchars @ + c!
      1 n-uchars +!
    then
  loop
  n-uchars @ to VOCAB_SIZE  \ note: +1 for BOS
  n-uchars @ to BOS_TOKEN
  VOCAB_SIZE 1+ to VOCAB_SIZE
  ." vocab size: " VOCAB_SIZE . cr ;

: char>tok ( c -- id )
  n-uchars @ 0 do
    dup uchars i + c@ = if drop i unloop exit then
  loop
  drop -1 ;

create tok-buf 32 cells allot

: tokenize ( doc-index -- n-tokens )
  { di }
  doc-idx di cells + @ { idx }
  BOS_TOKEN tok-buf !
  doc-len idx cells + @ { len }
  idx doc-addr { addr }
  len 0 do
    addr i + c@ char>tok  tok-buf i 1+ cells + !
  loop
  BOS_TOKEN tok-buf len 1+ cells + !
  len 2 + ;

\ ============================================================
\ Section 7: Parameter Initialization
\ ============================================================

\ Matrices stored as flat arrays of Value pointers (row-major)
\ mat[row][col] = mat + (row * cols + col) * cell

variable wte-mat      \ VOCAB_SIZE x N_EMBD
variable wpe-mat      \ BLOCK_SIZE x N_EMBD
variable lmhead-mat   \ VOCAB_SIZE x N_EMBD
variable attn-wq      \ N_EMBD x N_EMBD
variable attn-wk
variable attn-wv
variable attn-wo
variable mlp-fc1      \ (4*N_EMBD) x N_EMBD
variable mlp-fc2      \ N_EMBD x (4*N_EMBD)

: mat-el ( mat row col cols -- addr )
  rot * + cells + ;

: mat-create { rows cols -- addr }
  rows cols * cells allocate throw { m }
  rows 0 do
    cols 0 do
      0e 0.08e rng-gauss f* new-pval
      m j cols * i + cells + !
    loop
  loop
  m ;

\ Flat param list for optimizer
variable params-buf
variable n-params

: add-mat-params { mat n -- }
  n 0 do
    mat i cells + @
    params-buf @ n-params @ cells + !
    1 n-params +!
  loop ;

: init-params
  VOCAB_SIZE N_EMBD mat-create wte-mat !
  BLOCK_SIZE N_EMBD mat-create wpe-mat !
  VOCAB_SIZE N_EMBD mat-create lmhead-mat !
  N_EMBD N_EMBD mat-create attn-wq !
  N_EMBD N_EMBD mat-create attn-wk !
  N_EMBD N_EMBD mat-create attn-wv !
  N_EMBD N_EMBD mat-create attn-wo !
  N_EMBD 4 * N_EMBD mat-create mlp-fc1 !
  N_EMBD N_EMBD 4 * mat-create mlp-fc2 !
  \ Count and collect params
  VOCAB_SIZE N_EMBD *  BLOCK_SIZE N_EMBD * +  VOCAB_SIZE N_EMBD * +
  N_EMBD N_EMBD * 4 * +  N_EMBD 4 * N_EMBD * +  N_EMBD N_EMBD 4 * * +
  n-params !
  0 n-params !
  \ Allocate flat param list
  VOCAB_SIZE N_EMBD *  BLOCK_SIZE N_EMBD * +  VOCAB_SIZE N_EMBD * +
  N_EMBD N_EMBD * 4 * +  N_EMBD 4 * N_EMBD * +  N_EMBD N_EMBD 4 * * +
  cells allocate throw params-buf !
  wte-mat @    VOCAB_SIZE N_EMBD * add-mat-params
  wpe-mat @    BLOCK_SIZE N_EMBD * add-mat-params
  lmhead-mat @ VOCAB_SIZE N_EMBD * add-mat-params
  attn-wq @    N_EMBD N_EMBD * add-mat-params
  attn-wk @    N_EMBD N_EMBD * add-mat-params
  attn-wv @    N_EMBD N_EMBD * add-mat-params
  attn-wo @    N_EMBD N_EMBD * add-mat-params
  mlp-fc1 @    N_EMBD 4 * N_EMBD * add-mat-params
  mlp-fc2 @    N_EMBD N_EMBD 4 * * add-mat-params
  ." num params: " n-params @ . cr ;

\ ============================================================
\ Section 8: Neural Network Operations
\ ============================================================

\ linear(x, w) : x is array of `cols` Values, w is rows x cols matrix
\ Returns array of `rows` Values
: nn-linear { x w rows cols -- out }
  rows cells pool-alloc { out }
  rows 0 do
    i { row }
    \ dot product of w[row] and x
    w row cols * 0 + cells + @  x 0 cells + @  val*
    cols 1 do
      w row cols * i + cells + @  x i cells + @  val*  val+
    loop
    out row cells + !
  loop
  out ;

\ softmax(logits, n) : array of n Values -> array of n Values (probabilities)
fvariable sm-max

: nn-softmax { logits n -- probs }
  logits 0 cells + @ { l0 }
  l0 v.data f@ sm-max f!
  n 1 ?do
    logits i cells + @ v.data f@ sm-max f@ fmax sm-max f!
  loop
  \ exps[i] = exp(logits[i] - max)
  sm-max f@ fnegate new-val { negmax }
  n cells pool-alloc { exps }
  n 0 do
    logits i cells + @ negmax val+  val-exp
    exps i cells + !
  loop
  \ total = sum(exps)
  exps 0 cells + @
  n 1 ?do  exps i cells + @ val+  loop
  { total }
  \ probs = exps / total
  n cells pool-alloc { probs }
  n 0 do
    exps i cells + @ total val/
    probs i cells + !
  loop
  probs ;

\ rmsnorm(x, n) : array of n Values -> array of n Values
: nn-rmsnorm { x n -- out }
  \ ms = mean(x^2)
  x 0 cells + @ dup val*
  n 1 ?do  x i cells + @ dup val* val+  loop
  1e n s>f f/ new-val val*
  { ms }
  \ scale = (ms + eps)^-0.5
  ms 1e-5 new-val val+  -0.5e val**
  { scale }
  n cells pool-alloc { out }
  n 0 do
    x i cells + @ scale val*
    out i cells + !
  loop
  out ;

\ ============================================================
\ Section 9: GPT Forward Pass
\ ============================================================

\ KV cache: arrays of pointers to k/v vectors, one set per layer
\ For simplicity with N_LAYER=1, use flat arrays
create keys-cache BLOCK_SIZE cells allot
create vals-cache BLOCK_SIZE cells allot

: gpt-forward { tok-id pos-id -- logits }
  \ x = wte[tok_id] + wpe[pos_id]
  N_EMBD cells pool-alloc { x }
  N_EMBD 0 do
    wte-mat @ tok-id N_EMBD * i + cells + @
    wpe-mat @ pos-id N_EMBD * i + cells + @
    val+  x i cells + !
  loop
  \ x = rmsnorm(x)
  x N_EMBD nn-rmsnorm to x

  \ --- Layer 0 (only layer) ---
  \ Save residual
  x { xres }
  x N_EMBD nn-rmsnorm { xn }

  \ q, k, v projections
  xn attn-wq @ N_EMBD N_EMBD nn-linear { q }
  xn attn-wk @ N_EMBD N_EMBD nn-linear { k }
  xn attn-wv @ N_EMBD N_EMBD nn-linear { v }

  \ Store k, v in cache
  k keys-cache pos-id cells + !
  v vals-cache pos-id cells + !

  \ Multi-head attention
  N_EMBD cells pool-alloc { xattn }
  pos-id 1+ { npos }

  N_HEAD 0 do
    i HEAD_DIM * { hs }

    \ Compute attention logits for this head
    npos cells pool-alloc { alogits }
    npos 0 do
      \ dot(q[hs..], keys_cache[i][hs..]) / sqrt(HEAD_DIM)
      \ outer i = position, inner j = dim
      keys-cache i cells + @ { kt }
      q hs cells + @  kt hs cells + @  val*
      HEAD_DIM 1 do
        q hs i + cells + @  kt hs i + cells + @  val*  val+
      loop
      1e HEAD_DIM s>f fsqrt f/ new-val val*
      alogits i cells + !
    loop

    alogits npos nn-softmax { aw }

    \ head_out[d] = sum_t(aw[t] * vals_cache[t][hs+d])
    HEAD_DIM 0 do
      i { d }
      aw 0 cells + @
      vals-cache 0 cells + @ hs d + cells + @
      val*
      npos 1 ?do
        aw i cells + @
        vals-cache i cells + @ hs j + cells + @
        val*  val+
      loop
      xattn hs d + cells + !
    loop
  loop

  \ Project attention output
  xattn attn-wo @ N_EMBD N_EMBD nn-linear { xao }

  \ Residual connection
  N_EMBD cells pool-alloc { x2 }
  N_EMBD 0 do
    xao i cells + @  xres i cells + @  val+
    x2 i cells + !
  loop

  \ --- MLP block ---
  x2 { xres2 }
  x2 N_EMBD nn-rmsnorm { xn2 }

  xn2 mlp-fc1 @ N_EMBD 4 * N_EMBD nn-linear { h }
  \ ReLU
  N_EMBD 4 * 0 do
    h i cells + @ val-relu  h i cells + !
  loop
  h mlp-fc2 @ N_EMBD N_EMBD 4 * nn-linear { xm }

  \ Residual
  N_EMBD cells pool-alloc { x3 }
  N_EMBD 0 do
    xm i cells + @  xres2 i cells + @  val+
    x3 i cells + !
  loop

  \ Output logits
  x3 lmhead-mat @ VOCAB_SIZE N_EMBD nn-linear ;

\ ============================================================
\ Section 10: Adam Optimizer
\ ============================================================

variable adam-m   \ first moment buffer (floats)
variable adam-v   \ second moment buffer (floats)

0.01e  fconstant LR
0.85e  fconstant BETA1
0.99e  fconstant BETA2
1e-8   fconstant EPS_ADAM

: init-adam
  n-params @ floats allocate throw adam-m !
  n-params @ floats allocate throw adam-v !
  n-params @ 0 do
    0e adam-m @ i floats + f!
    0e adam-v @ i floats + f!
  loop ;

: adam-update { step -- }
  LR 1e step s>f NUM_STEPS s>f f/ f- f* { F: lrt }
  step 1+ s>f { F: t }
  n-params @ 0 do
    params-buf @ i cells + @ { p }
    p v.grad f@ { F: g }
    \ m = beta1*m + (1-beta1)*g
    adam-m @ i floats + f@ BETA1 f*  1e BETA1 f- g f* f+  { F: mi }
    mi adam-m @ i floats + f!
    \ v = beta2*v + (1-beta2)*g^2
    adam-v @ i floats + f@ BETA2 f*  1e BETA2 f- g g f* f* f+  { F: vi }
    vi adam-v @ i floats + f!
    \ bias correction
    mi 1e BETA1 t f** f- f/  { F: mh }
    vi 1e BETA2 t f** f- f/  { F: vh }
    \ update
    p v.data f@  lrt mh f* vh fsqrt EPS_ADAM f+ f/ f-  p v.data f!
    0e p v.grad f!
  loop ;

\ ============================================================
\ Section 11: Training Loop
\ ============================================================

: train-step { step -- }
  reset-pool
  \ Tokenize document
  step n-docs @ mod tokenize { ntok }
  BLOCK_SIZE ntok 1- min { n }

  \ Forward pass
  n cells pool-alloc { losses }
  n 0 do
    tok-buf i cells + @ { tok }
    tok-buf i 1+ cells + @ { target }
    tok i gpt-forward { logits }
    logits VOCAB_SIZE nn-softmax { probs }
    probs target cells + @ val-log val-neg
    losses i cells + !
  loop

  \ Average loss
  losses 0 cells + @
  n 1 ?do  losses i cells + @ val+  loop
  1e n s>f f/ new-val val*
  { loss }

  \ Backward
  loss backward

  \ Adam update
  step adam-update

  \ Print
  ." step " step 1+ 4 u.r ."  / " NUM_STEPS 4 u.r
  ."  | loss " loss v.data f@ f. cr ;

: train
  NUM_STEPS 0 do
    i train-step
  loop ;

\ ============================================================
\ Section 12: Inference
\ ============================================================

0.5e fconstant TEMPERATURE

: sample-token { probs -- tok }
  \ Weighted random selection from probability array
  rng-float { F: r }
  VOCAB_SIZE 0 do
    r probs i cells + @ v.data f@ f-  { F: r2 }  r2 to r
    r f0< if  i  unloop exit  then
  loop
  VOCAB_SIZE 1- ;

: inference
  cr ." --- inference (new, hallucinated names) ---" cr
  20 0 do
    reset-pool
    BOS_TOKEN { tok }
    pad 0 { buf pos }
    BLOCK_SIZE 0 do
      \ Apply temperature: divide logits by temperature
      tok i gpt-forward { logits }
      VOCAB_SIZE cells pool-alloc { scaled }
      VOCAB_SIZE 0 do
        logits i cells + @
        TEMPERATURE new-val val/
        scaled i cells + !
      loop
      scaled VOCAB_SIZE nn-softmax { probs }
      probs sample-token to tok
      tok BOS_TOKEN = if  leave  then
      uchars tok + c@ buf pos + c!
      pos 1+ to pos
    loop
    ." sample " i 1+ 2 u.r ." : "
    buf pos type cr
  loop ;

\ ============================================================
\ Section 13: Main
\ ============================================================

: main
  init-pool
  load-data
  init-indices
  shuffle-indices
  build-vocab
  init-params
  init-adam
  train
  inference
  bye ;

main
