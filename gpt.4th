\ ============================================================
\ GPT in Forth (gforth) — port of Karpathy's atomic GPT
\ ============================================================
\ Usage: gforth gpt.4th

\ ============================================================
\ Configuration
\ ============================================================
1 constant N_LAYER
16 constant N_EMBD
16 constant BLOCK_SIZE
4 constant N_HEAD
N_EMBD N_HEAD / constant HEAD_DIM
200 constant NUM_STEPS
257 constant MAX_VOCAB
200000 constant MAX_DOCS
300000 constant MAX_NODES

\ ============================================================
\ RNG (LCG + Box-Muller)
\ ============================================================
variable rng-state  42 rng-state !
: rng ( -- u ) rng-state @ 1103515245 * 12345 + 2147483647 and dup rng-state ! ;
: rng-mod ( n -- r ) rng swap mod abs ;
: rng-float ( -- ) ( F: -- r ) rng s>f 2147483647e0 f/ ;
3.14159265358979e0 fconstant F_PI
: rng-gauss ( -- ) ( F: std -- val )
  begin rng-float fdup f0= 0= until
  rng-float 2e0 F_PI f* f* fcos
  fswap fln -2e0 f* fsqrt f* f* ;

\ ============================================================
\ Node Pool (autograd values)
\ ============================================================
MAX_NODES floats allocate throw constant nd-data
MAX_NODES floats allocate throw constant nd-grad
MAX_NODES cells  allocate throw constant nd-nch
MAX_NODES cells  allocate throw constant nd-ch1
MAX_NODES cells  allocate throw constant nd-ch2
MAX_NODES floats allocate throw constant nd-lg1
MAX_NODES floats allocate throw constant nd-lg2
variable nodes-used  0 nodes-used !
variable params-end

\ Create a new Value node
: val ( -- idx ) ( F: data -- )
  nodes-used @ dup MAX_NODES >= abort" Node pool full"
  dup floats nd-data + f!
  0e0 over floats nd-grad + f!
  0 over cells nd-nch + !
  0e0 over floats nd-lg1 + f!
  0e0 over floats nd-lg2 + f!
  nodes-used @ 1+ nodes-used ! ;

\ Accessors
: nd@ ( idx -- ) ( F: -- v ) floats nd-data + f@ ;
: ng@ ( idx -- ) ( F: -- v ) floats nd-grad + f@ ;
: ng! ( idx -- ) ( F: v -- ) floats nd-grad + f! ;

: reset-nodes ( -- )
  params-end @ nodes-used !
  params-end @ 0 ?do 0e0 i ng! loop ;

\ ============================================================
\ Autograd Operations
\ ============================================================

\ add: c = a + b
: val+ ( a b -- c )
  over nd@ dup nd@ f+ val
  2 over cells nd-nch + !
  rot over cells nd-ch1 + !
  swap over cells nd-ch2 + !
  1e0 over floats nd-lg1 + f!
  1e0 over floats nd-lg2 + f! ;

\ multiply: c = a * b, dloss/da += b_data * grad, dloss/db += a_data * grad
: val* ( a b -- c )
  over nd@ dup nd@
  fover fover f* val
  2 over cells nd-nch + !
  rot over cells nd-ch1 + !
  swap over cells nd-ch2 + !
  dup floats nd-lg1 + f!
  dup floats nd-lg2 + f! ;

\ power: c = a^exp (exp is a float constant)
: val** ( a -- c ) ( F: exp -- )
  fdup dup nd@ fswap f** val
  1 over cells nd-nch + !
  swap over cells nd-ch1 + !
  dup cells nd-ch1 + @ nd@
  fswap fdup frot fswap 1e0 f- f** f*
  dup floats nd-lg1 + f! ;

\ exp: c = e^a
: val-exp ( a -- c )
  dup nd@ fexp fdup val
  1 over cells nd-nch + !
  swap over cells nd-ch1 + !
  dup floats nd-lg1 + f! ;

\ log: c = ln(a)
: val-log ( a -- c )
  dup nd@ fdup fln val
  1 over cells nd-nch + !
  swap over cells nd-ch1 + !
  1e0 fswap f/
  dup floats nd-lg1 + f! ;

\ relu: c = max(0, a)
: val-relu ( a -- c )
  dup nd@
  fdup f0< fdup f0= or if
    fdrop 0e0 val
    1 over cells nd-nch + !
    swap over cells nd-ch1 + !
    0e0 over floats nd-lg1 + f!
  else
    val
    1 over cells nd-nch + !
    swap over cells nd-ch1 + !
    1e0 over floats nd-lg1 + f!
  then ;

: val-neg ( a -- c ) -1e0 val swap val* ;
: val-sub ( a b -- c ) val-neg val+ ;
: val-div ( a b -- c ) -1e0 val** val* ;

\ ============================================================
\ Backward Pass (iterative topo sort)
\ ============================================================
MAX_NODES allocate throw constant topo-vis
MAX_NODES cells allocate throw constant topo-ord
variable topo-cnt
MAX_NODES 3 * cells allocate throw constant dfs-stk
variable dfs-sp
variable bk-node

: backward ( loss-idx -- )
  \ Reset
  topo-vis MAX_NODES erase
  0 topo-cnt !
  0 dfs-sp !
  \ Push root in visit phase (encoded as idx*2)
  dup 2* dfs-sp @ cells dfs-stk + ! dfs-sp @ 1+ dfs-sp !
  \ Build topo order iteratively
  begin dfs-sp @ 0> while
    dfs-sp @ 1- dup dfs-sp ! cells dfs-stk + @
    dup 1 and if
      \ Finalize phase: add to topo
      2/
      dup topo-cnt @ cells topo-ord + !
      topo-cnt @ 1+ topo-cnt !
      drop
    else
      2/
      dup topo-vis + c@ if
        drop
      else
        dup 1 swap topo-vis + c!
        \ Push finalize
        dup 2* 1+  dfs-sp @ cells dfs-stk + ! dfs-sp @ 1+ dfs-sp !
        \ Push child2 if exists
        dup cells nd-nch + @ 1 > if
          dup cells nd-ch2 + @
          dup topo-vis + c@ 0= if
            2*  dfs-sp @ cells dfs-stk + ! dfs-sp @ 1+ dfs-sp !
          else drop then
        then
        \ Push child1 if exists
        dup cells nd-nch + @ 0> if
          dup cells nd-ch1 + @
          dup topo-vis + c@ 0= if
            2*  dfs-sp @ cells dfs-stk + ! dfs-sp @ 1+ dfs-sp !
          else drop then
        then
        drop
      then
    then
  repeat
  \ Set loss gradient to 1
  1e0 swap ng!
  \ Propagate gradients in reverse topo order
  topo-cnt @ 0 ?do
    topo-cnt @ 1- i - cells topo-ord + @ bk-node !
    bk-node @ ng@
    bk-node @ cells nd-nch + @ dup 0> if
      bk-node @ cells nd-ch1 + @
      bk-node @ floats nd-lg1 + f@
      fover f*
      dup ng@ f+
      dup ng!
      drop
    then
    dup 1 > if
      bk-node @ cells nd-ch2 + @
      bk-node @ floats nd-lg2 + f@
      fover f*
      dup ng@ f+
      dup ng!
      drop
    then
    drop fdrop
  loop ;

\ ============================================================
\ Dataset
\ ============================================================
8000000 allocate throw constant text-buf
variable text-ptr  0 text-ptr !
MAX_DOCS cells allocate throw constant doc-addr
MAX_DOCS cells allocate throw constant doc-len
variable num-docs  0 num-docs !
variable input-fid
variable tmp-swap

create char-flag 256 allot
create char-map  256 cells allot
create id-map    256 cells allot
variable vocab-size  0 vocab-size !
variable bos-token

: load-data ( -- )
  s" input.txt" r/o open-file throw input-fid !
  begin
    pad 512 input-fid @ read-line throw
  while
    dup 0> if
      text-buf text-ptr @ +
      dup num-docs @ cells doc-addr + !
      over num-docs @ cells doc-len + !
      pad swap 2 pick swap move
      text-ptr @ + text-ptr !
      num-docs @ 1+ num-docs !
    else drop then
  repeat
  drop
  input-fid @ close-file throw
  ." num docs: " num-docs @ . cr ;

: build-vocab ( -- )
  char-flag 256 erase
  num-docs @ 0 ?do
    i cells doc-addr + @
    i cells doc-len + @
    0 ?do
      dup i + c@
      char-flag + 1 swap c!
    loop
    drop
  loop
  0 vocab-size !
  256 0 do
    char-flag i + c@ if
      vocab-size @ i cells char-map + !
      i vocab-size @ cells id-map + !
      vocab-size @ 1+ vocab-size !
    then
  loop
  vocab-size @ bos-token !
  vocab-size @ 1+ vocab-size !
  ." vocab size: " vocab-size @ . cr ;

: shuffle-docs ( -- )
  num-docs @ 1 ?do
    i 1+ rng-mod
    dup cells doc-addr + @ tmp-swap !
    i cells doc-addr + @ over cells doc-addr + !
    tmp-swap @ i cells doc-addr + !
    dup cells doc-len + @ tmp-swap !
    i cells doc-len + @ over cells doc-len + !
    tmp-swap @ i cells doc-len + !
    drop
  loop ;

\ ============================================================
\ Model Parameters
\ ============================================================
variable wte-ptr
variable wpe-ptr
variable lmhead-ptr
variable wq-ptr
variable wk-ptr
variable wv-ptr
variable wo-ptr
variable fc1-ptr
variable fc2-ptr

\ Adam optimizer buffers
variable adam-m
variable adam-v
variable num-params

: alloc-matrix ( nrows ncols -- addr )
  * cells allocate throw ;

: init-matrix ( addr nrows ncols -- ) ( F: std -- )
  * 0 ?do
    fdup rng-gauss val
    over i cells + !
  loop fdrop drop ;

: init-model ( -- )
  \ Allocate matrices
  vocab-size @ N_EMBD alloc-matrix wte-ptr !
  BLOCK_SIZE N_EMBD alloc-matrix wpe-ptr !
  vocab-size @ N_EMBD alloc-matrix lmhead-ptr !
  N_EMBD N_EMBD alloc-matrix wq-ptr !
  N_EMBD N_EMBD alloc-matrix wk-ptr !
  N_EMBD N_EMBD alloc-matrix wv-ptr !
  N_EMBD N_EMBD alloc-matrix wo-ptr !
  N_EMBD 4 * N_EMBD alloc-matrix fc1-ptr !
  N_EMBD N_EMBD 4 * alloc-matrix fc2-ptr !
  \ Initialize with random weights
  0.08e0
  wte-ptr @ vocab-size @ N_EMBD fdup init-matrix
  wpe-ptr @ BLOCK_SIZE N_EMBD fdup init-matrix
  lmhead-ptr @ vocab-size @ N_EMBD fdup init-matrix
  wq-ptr @ N_EMBD N_EMBD fdup init-matrix
  wk-ptr @ N_EMBD N_EMBD fdup init-matrix
  wv-ptr @ N_EMBD N_EMBD fdup init-matrix
  wo-ptr @ N_EMBD N_EMBD fdup init-matrix
  fc1-ptr @ N_EMBD 4 * N_EMBD fdup init-matrix
  fc2-ptr @ N_EMBD N_EMBD 4 * fdup init-matrix
  fdrop
  \ Mark end of parameter nodes
  nodes-used @ params-end !
  nodes-used @ num-params !
  ." num params: " num-params @ . cr
  \ Adam buffers (plain floats, not Value nodes)
  num-params @ floats allocate throw adam-m !
  num-params @ floats allocate throw adam-v !
  adam-m @ num-params @ floats erase
  adam-v @ num-params @ floats erase ;

\ ============================================================
\ NN Operations
\ ============================================================

\ Buffers for forward pass
create buf-x     N_EMBD cells allot
create buf-xres  N_EMBD cells allot
create buf-q     N_EMBD cells allot
create buf-k     N_EMBD cells allot
create buf-v     N_EMBD cells allot
create buf-xattn N_EMBD cells allot
create buf-mlp   N_EMBD 4 * cells allot
MAX_VOCAB cells allocate throw constant buf-logits
create buf-al    BLOCK_SIZE cells allot
create buf-tmp   N_EMBD cells allot

\ KV cache
N_LAYER BLOCK_SIZE * N_EMBD * cells allocate throw constant kv-keys
N_LAYER BLOCK_SIZE * N_EMBD * cells allocate throw constant kv-vals

variable v-li    \ current layer
variable v-pos   \ current position
variable v-nkv   \ num key-value pairs

: buf-copy ( src dst n -- )
  0 ?do
    over i cells + @
    over i cells + !
  loop 2drop ;

\ linear: y[r] = sum_c(w[r*nin+c] * x[c])
: nn-linear ( x w nout nin y -- )
  >r >r >r >r >r  \ stash all: R: y nin nout w x
  r> r> r> r> r>   \ restore: x w nout nin y
  { ybuf nin2 nout2 wbuf xbuf }
  nout2 0 ?do
    wbuf i nin2 * cells + @ xbuf @ val*
    nin2 1 ?do
      wbuf j nin2 * i + cells + @ xbuf i cells + @ val* val+
    loop
    ybuf i cells + !
  loop ;

\ softmax in-place on buffer of n node indices
: nn-softmax ( buf n -- )
  { n2 sbuf }
  sbuf @ nd@
  n2 1 ?do
    sbuf i cells + @ nd@
    fover fover f< if fswap fdrop else fdrop then
  loop
  val { maxn }
  n2 0 ?do
    sbuf i cells + @ maxn val-sub val-exp
    sbuf i cells + !
  loop
  sbuf @
  n2 1 ?do sbuf i cells + @ val+ loop
  { totn }
  n2 0 ?do
    sbuf i cells + @ totn val-div
    sbuf i cells + !
  loop ;

\ rmsnorm in-place
: nn-rmsnorm ( buf n -- )
  { n3 rbuf }
  rbuf @ dup val*
  n3 1 ?do rbuf i cells + @ dup val* val+ loop
  n3 s>f val val-div
  1e-5 val val+
  -0.5e0 val**
  { scl }
  n3 0 ?do
    rbuf i cells + @ scl val*
    rbuf i cells + !
  loop ;

\ ============================================================
\ GPT Forward Pass
\ ============================================================

\ Store key/value vector into KV cache
: store-kv ( layer pos src-buf base -- )
  { kvbase }
  { srcb posv layv }
  N_EMBD 0 ?do
    srcb i cells + @
    kvbase layv BLOCK_SIZE N_EMBD * * posv N_EMBD * + i + cells + !
  loop ;

\ Attention for one head
variable v-head

: compute-attn ( -- )
  v-head @ HEAD_DIM * { hs }
  v-nkv @ 0 ?do
    \ dot product q_h . k_h[t] for time step i
    buf-q hs cells + @
    kv-keys v-li @ BLOCK_SIZE N_EMBD * * i N_EMBD * + hs + cells + @
    val*
    HEAD_DIM 1 ?do
      buf-q hs j + cells + @
      kv-keys v-li @ BLOCK_SIZE N_EMBD * * j N_EMBD * + hs i + cells + @
      val*
      val+
    loop
    \ Scale by 1/sqrt(head_dim)
    HEAD_DIM s>f fsqrt 1e0 fswap f/ val val*
    buf-al i cells + !
  loop
  \ Softmax
  buf-al v-nkv @ nn-softmax
  \ Weighted sum of values
  HEAD_DIM 0 ?do
    \ sum_t attn[t] * v[t][hs+j] for dimension i (outer), t (inner)
    buf-al @
    kv-vals v-li @ BLOCK_SIZE N_EMBD * * 0 N_EMBD * + hs i + cells + @
    val*
    v-nkv @ 1 ?do
      buf-al i cells + @
      kv-vals v-li @ BLOCK_SIZE N_EMBD * * i N_EMBD * + hs j + cells + @
      val* val+
    loop
    buf-xattn hs i + cells + !
  loop ;

: gpt-forward ( tok pos -- )
  dup v-pos ! swap
  \ Token embedding
  N_EMBD 0 ?do
    wte-ptr @ over N_EMBD * i + cells + @
    buf-x i cells + !
  loop drop
  \ Position embedding (add)
  N_EMBD 0 ?do
    buf-x i cells + @
    wpe-ptr @ v-pos @ N_EMBD * i + cells + @
    val+
    buf-x i cells + !
  loop
  \ Initial RMSNorm
  buf-x N_EMBD nn-rmsnorm
  \ Transformer layers
  N_LAYER 0 ?do
    i v-li !
    \ Save residual
    buf-x buf-xres N_EMBD buf-copy
    \ RMSNorm
    buf-x N_EMBD nn-rmsnorm
    \ Q K V projections
    buf-x wq-ptr @ N_EMBD N_EMBD buf-q nn-linear
    buf-x wk-ptr @ N_EMBD N_EMBD buf-k nn-linear
    buf-x wv-ptr @ N_EMBD N_EMBD buf-v nn-linear
    \ Store K,V in cache
    v-li @ v-pos @ buf-k kv-keys store-kv
    v-li @ v-pos @ buf-v kv-vals store-kv
    v-pos @ 1+ v-nkv !
    \ Multi-head attention
    N_HEAD 0 ?do
      i v-head !
      compute-attn
    loop
    \ Output projection
    buf-xattn wo-ptr @ N_EMBD N_EMBD buf-x nn-linear
    \ Residual
    N_EMBD 0 ?do
      buf-x i cells + @ buf-xres i cells + @ val+
      buf-x i cells + !
    loop
    \ MLP
    buf-x buf-xres N_EMBD buf-copy
    buf-x N_EMBD nn-rmsnorm
    buf-x fc1-ptr @ N_EMBD 4 * N_EMBD buf-mlp nn-linear
    N_EMBD 4 * 0 ?do
      buf-mlp i cells + @ val-relu buf-mlp i cells + !
    loop
    buf-mlp fc2-ptr @ N_EMBD N_EMBD 4 * buf-x nn-linear
    N_EMBD 0 ?do
      buf-x i cells + @ buf-xres i cells + @ val+
      buf-x i cells + !
    loop
  loop
  \ LM head
  buf-x lmhead-ptr @ vocab-size @ N_EMBD buf-logits nn-linear ;

\ ============================================================
\ Training
\ ============================================================

create tok-buf 520 cells allot
variable loss-node

: train ( -- )
  0.01e0 { f: lr }
  NUM_STEPS 0 ?do
    reset-nodes
    \ Pick document
    i num-docs @ mod cells doc-addr + @ { daddr }
    i num-docs @ mod cells doc-len + @ { dlen }
    \ Tokenize: [BOS] + chars + [BOS]
    bos-token @ tok-buf !
    dlen BLOCK_SIZE 1- min { tlen }
    tlen 0 ?do
      daddr i + c@ cells char-map + @
      tok-buf i 1+ cells + !
    loop
    bos-token @ tok-buf tlen 1+ cells + !
    tlen 1+ { seqlen }
    \ Forward pass
    seqlen 0 ?do
      tok-buf i cells + @
      i
      gpt-forward
      \ Softmax on logits
      buf-logits vocab-size @ nn-softmax
      \ Loss: -log(prob[target])
      tok-buf i 1+ cells + @ cells buf-logits + @
      val-log val-neg
    loop
    \ Average losses: the loop left seqlen loss nodes on stack
    seqlen 1 ?do val+ loop
    seqlen s>f 1e0 fswap f/ val val*
    dup loss-node !
    \ Backward
    backward
    \ Adam update
    lr 1e0 i s>f NUM_STEPS s>f f/ f- f*
    num-params @ 0 ?do
      i floats nd-grad + f@ { f: gi }
      adam-m @ i floats +
      dup f@ 0.85e0 f* gi 0.15e0 f* f+ dup fdup f!
      f@ { f: mh }
      adam-v @ i floats +
      dup f@ 0.99e0 f* gi gi f* 0.01e0 f* f+ dup fdup f!
      f@ { f: vh }
      mh 1e0 0.85e0 j 1+ s>f f** f- f/
      vh 1e0 0.99e0 j 1+ s>f f** f- f/
      fsqrt 1e-8 f+ f/
      fover f* fnegate
      i floats nd-data + dup f@ fswap f+ f!
    loop
    fdrop
    \ Print progress
    ." step " j 1+ 4 .r ."  / " NUM_STEPS 4 .r ."  | loss "
    loss-node @ nd@ f. cr
  loop ;

\ ============================================================
\ Inference
\ ============================================================

: generate ( -- )
  0.5e0 { f: temp }
  ." --- inference (sacred text hallucinations) ---" cr
  20 0 ?do
    \ Reset computation nodes for each sample
    params-end @ nodes-used !
    bos-token @ { curtok }
    ." sample " i 1+ 2 .r ." : "
    BLOCK_SIZE 0 ?do
      curtok i gpt-forward
      \ Temperature-scaled softmax
      vocab-size @ 0 ?do
        buf-logits i cells + @ nd@
        temp f/
        val
        buf-logits i cells + !
      loop
      buf-logits vocab-size @ nn-softmax
      \ Sample from distribution
      rng-float { f: r }
      0e0 { f: cumul }
      0 { chosen }
      vocab-size @ 0 ?do
        buf-logits i cells + @ nd@
        cumul f+ to cumul
        cumul r f> if
          i to chosen
          leave
        then
      loop
      chosen bos-token @ = if leave then
      chosen cells id-map + @ emit
      chosen to curtok
    loop
    cr
  loop ;

\ ============================================================
\ Main
\ ============================================================
: main ( -- )
  ." Loading dataset..." cr
  load-data
  build-vocab
  shuffle-docs
  ." Initializing model..." cr
  init-model
  ." Training..." cr
  train
  cr
  generate
  bye ;

main
