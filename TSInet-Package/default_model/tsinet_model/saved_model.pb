๐(
ัฃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
พ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878มข$
{
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
t
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*#
_output_shapes
:*
dtype0
o
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
h
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes	
:*
dtype0

conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_1/kernel
y
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*$
_output_shapes
:*
dtype0
s
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
l
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes	
:*
dtype0

conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1d_2/kernel
y
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*$
_output_shapes
:*
dtype0
s
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
l
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes	
:*
dtype0
ด
,seq_self_attention/seq_self_attention_Add_WtVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *=
shared_name.,seq_self_attention/seq_self_attention_Add_Wt
ญ
@seq_self_attention/seq_self_attention_Add_Wt/Read/ReadVariableOpReadVariableOp,seq_self_attention/seq_self_attention_Add_Wt*
_output_shapes

:
 *
dtype0
ด
,seq_self_attention/seq_self_attention_Add_WxVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *=
shared_name.,seq_self_attention/seq_self_attention_Add_Wx
ญ
@seq_self_attention/seq_self_attention_Add_Wx/Read/ReadVariableOpReadVariableOp,seq_self_attention/seq_self_attention_Add_Wx*
_output_shapes

:
 *
dtype0
ฐ
,seq_self_attention/seq_self_attention_Add_bhVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,seq_self_attention/seq_self_attention_Add_bh
ฉ
@seq_self_attention/seq_self_attention_Add_bh/Read/ReadVariableOpReadVariableOp,seq_self_attention/seq_self_attention_Add_bh*
_output_shapes
: *
dtype0
ด
,seq_self_attention/seq_self_attention_Add_WaVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,seq_self_attention/seq_self_attention_Add_Wa
ญ
@seq_self_attention/seq_self_attention_Add_Wa/Read/ReadVariableOpReadVariableOp,seq_self_attention/seq_self_attention_Add_Wa*
_output_shapes

: *
dtype0
ฐ
,seq_self_attention/seq_self_attention_Add_baVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,seq_self_attention/seq_self_attention_Add_ba
ฉ
@seq_self_attention/seq_self_attention_Add_ba/Read/ReadVariableOpReadVariableOp,seq_self_attention/seq_self_attention_Add_ba*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*&
shared_namelstm/lstm_cell/kernel

)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:	(*
dtype0

lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*0
shared_name!lstm/lstm_cell/recurrent_kernel

3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes

:
(*
dtype0
~
lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_namelstm/lstm_cell/bias
w
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes
:(*
dtype0

time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*(
shared_nametime_distributed/kernel

+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel*
_output_shapes
:	
*
dtype0

time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nametime_distributed/bias
|
)time_distributed/bias/Read/ReadVariableOpReadVariableOptime_distributed/bias*
_output_shapes	
:*
dtype0

time_distributed_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nametime_distributed_1/kernel

-time_distributed_1/kernel/Read/ReadVariableOpReadVariableOptime_distributed_1/kernel* 
_output_shapes
:
*
dtype0

time_distributed_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed_1/bias

+time_distributed_1/bias/Read/ReadVariableOpReadVariableOptime_distributed_1/bias*
_output_shapes	
:*
dtype0

time_distributed_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nametime_distributed_2/kernel

-time_distributed_2/kernel/Read/ReadVariableOpReadVariableOptime_distributed_2/kernel*
_output_shapes
:	*
dtype0

time_distributed_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed_2/bias

+time_distributed_2/bias/Read/ReadVariableOpReadVariableOptime_distributed_2/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d/kernel/m

(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*#
_output_shapes
:*
dtype0
}
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/m
v
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes	
:*
dtype0

Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_1/kernel/m

*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*$
_output_shapes
:*
dtype0

Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/m
z
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes	
:*
dtype0

Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_2/kernel/m

*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*$
_output_shapes
:*
dtype0

Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_2/bias/m
z
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes	
:*
dtype0
ย
3Adam/seq_self_attention/seq_self_attention_Add_Wt/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_Wt/m
ป
GAdam/seq_self_attention/seq_self_attention_Add_Wt/m/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_Wt/m*
_output_shapes

:
 *
dtype0
ย
3Adam/seq_self_attention/seq_self_attention_Add_Wx/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_Wx/m
ป
GAdam/seq_self_attention/seq_self_attention_Add_Wx/m/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_Wx/m*
_output_shapes

:
 *
dtype0
พ
3Adam/seq_self_attention/seq_self_attention_Add_bh/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_bh/m
ท
GAdam/seq_self_attention/seq_self_attention_Add_bh/m/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_bh/m*
_output_shapes
: *
dtype0
ย
3Adam/seq_self_attention/seq_self_attention_Add_Wa/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_Wa/m
ป
GAdam/seq_self_attention/seq_self_attention_Add_Wa/m/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_Wa/m*
_output_shapes

: *
dtype0
พ
3Adam/seq_self_attention/seq_self_attention_Add_ba/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_ba/m
ท
GAdam/seq_self_attention/seq_self_attention_Add_ba/m/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_ba/m*
_output_shapes
:*
dtype0

Adam/lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*-
shared_nameAdam/lstm/lstm_cell/kernel/m

0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/m*
_output_shapes
:	(*
dtype0
จ
&Adam/lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/m
ก
:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/m*
_output_shapes

:
(*
dtype0

Adam/lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*+
shared_nameAdam/lstm/lstm_cell/bias/m

.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/m*
_output_shapes
:(*
dtype0

Adam/time_distributed/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*/
shared_name Adam/time_distributed/kernel/m

2Adam/time_distributed/kernel/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed/kernel/m*
_output_shapes
:	
*
dtype0

Adam/time_distributed/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/time_distributed/bias/m

0Adam/time_distributed/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed/bias/m*
_output_shapes	
:*
dtype0

 Adam/time_distributed_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/time_distributed_1/kernel/m

4Adam/time_distributed_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/time_distributed_1/kernel/m* 
_output_shapes
:
*
dtype0

Adam/time_distributed_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/time_distributed_1/bias/m

2Adam/time_distributed_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed_1/bias/m*
_output_shapes	
:*
dtype0

 Adam/time_distributed_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/time_distributed_2/kernel/m

4Adam/time_distributed_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/time_distributed_2/kernel/m*
_output_shapes
:	*
dtype0

Adam/time_distributed_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/time_distributed_2/bias/m

2Adam/time_distributed_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed_2/bias/m*
_output_shapes
:*
dtype0

Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d/kernel/v

(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*#
_output_shapes
:*
dtype0
}
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/v
v
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes	
:*
dtype0

Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_1/kernel/v

*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*$
_output_shapes
:*
dtype0

Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/v
z
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes	
:*
dtype0

Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1d_2/kernel/v

*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*$
_output_shapes
:*
dtype0

Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_2/bias/v
z
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes	
:*
dtype0
ย
3Adam/seq_self_attention/seq_self_attention_Add_Wt/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_Wt/v
ป
GAdam/seq_self_attention/seq_self_attention_Add_Wt/v/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_Wt/v*
_output_shapes

:
 *
dtype0
ย
3Adam/seq_self_attention/seq_self_attention_Add_Wx/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_Wx/v
ป
GAdam/seq_self_attention/seq_self_attention_Add_Wx/v/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_Wx/v*
_output_shapes

:
 *
dtype0
พ
3Adam/seq_self_attention/seq_self_attention_Add_bh/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_bh/v
ท
GAdam/seq_self_attention/seq_self_attention_Add_bh/v/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_bh/v*
_output_shapes
: *
dtype0
ย
3Adam/seq_self_attention/seq_self_attention_Add_Wa/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_Wa/v
ป
GAdam/seq_self_attention/seq_self_attention_Add_Wa/v/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_Wa/v*
_output_shapes

: *
dtype0
พ
3Adam/seq_self_attention/seq_self_attention_Add_ba/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/seq_self_attention/seq_self_attention_Add_ba/v
ท
GAdam/seq_self_attention/seq_self_attention_Add_ba/v/Read/ReadVariableOpReadVariableOp3Adam/seq_self_attention/seq_self_attention_Add_ba/v*
_output_shapes
:*
dtype0

Adam/lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*-
shared_nameAdam/lstm/lstm_cell/kernel/v

0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/v*
_output_shapes
:	(*
dtype0
จ
&Adam/lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/v
ก
:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/v*
_output_shapes

:
(*
dtype0

Adam/lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*+
shared_nameAdam/lstm/lstm_cell/bias/v

.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/v*
_output_shapes
:(*
dtype0

Adam/time_distributed/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*/
shared_name Adam/time_distributed/kernel/v

2Adam/time_distributed/kernel/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed/kernel/v*
_output_shapes
:	
*
dtype0

Adam/time_distributed/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/time_distributed/bias/v

0Adam/time_distributed/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed/bias/v*
_output_shapes	
:*
dtype0

 Adam/time_distributed_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/time_distributed_1/kernel/v

4Adam/time_distributed_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/time_distributed_1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/time_distributed_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/time_distributed_1/bias/v

2Adam/time_distributed_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed_1/bias/v*
_output_shapes	
:*
dtype0

 Adam/time_distributed_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/time_distributed_2/kernel/v

4Adam/time_distributed_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/time_distributed_2/kernel/v*
_output_shapes
:	*
dtype0

Adam/time_distributed_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/time_distributed_2/bias/v

2Adam/time_distributed_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ขv
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?u
valueำuBะu Bษu
ั
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
R
$trainable_variables
%regularization_losses
&	variables
'	keras_api
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
l
0cell
1
state_spec
2regularization_losses
3trainable_variables
4	variables
5	keras_api

6seq_self_attention_Add_Wt
6Wt
7seq_self_attention_Add_Wx
7Wx
8seq_self_attention_Add_bh
8bh
9seq_self_attention_Add_Wa
9Wa
:seq_self_attention_Add_ba
:ba
;trainable_variables
<regularization_losses
=	variables
>	keras_api
]
	?layer
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
]
	Dlayer
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
]
	Ilayer
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
ะ
Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_ratemยmรmฤmลmฦmว6mศ7mษ8mส9mห:mฬSmอTmฮUmฯVmะWmัXmาYmำZmิ[mีvึvืvุvูvฺv?6v?7v?8v?9v฿:vเSvแTvโUvใVvไWvๅXvๆYv็Zv่[v้
 

0
1
2
3
4
5
S6
T7
U8
69
710
811
912
:13
V14
W15
X16
Y17
Z18
[19

0
1
2
3
4
5
S6
T7
U8
69
710
811
912
:13
V14
W15
X16
Y17
Z18
[19
ญ
\layer_regularization_losses
]non_trainable_variables
regularization_losses
trainable_variables
^layer_metrics

_layers
`metrics
	variables
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
ญ
alayer_regularization_losses
bnon_trainable_variables
trainable_variables
regularization_losses
clayer_metrics

dlayers
emetrics
	variables
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
ญ
flayer_regularization_losses
gnon_trainable_variables
trainable_variables
regularization_losses
hlayer_metrics

ilayers
jmetrics
	variables
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
ญ
klayer_regularization_losses
lnon_trainable_variables
 trainable_variables
!regularization_losses
mlayer_metrics

nlayers
ometrics
"	variables
 
 
 
ญ
player_regularization_losses
qnon_trainable_variables
$trainable_variables
%regularization_losses
rlayer_metrics

slayers
tmetrics
&	variables
 
 
 
ญ
ulayer_regularization_losses
vnon_trainable_variables
(trainable_variables
)regularization_losses
wlayer_metrics

xlayers
ymetrics
*	variables
 
 
 
ญ
zlayer_regularization_losses
{non_trainable_variables
,trainable_variables
-regularization_losses
|layer_metrics

}layers
~metrics
.	variables


Skernel
Trecurrent_kernel
Ubias
trainable_variables
regularization_losses
	variables
	keras_api
 
 

S0
T1
U2

S0
T1
U2
ฟ
states
 layer_regularization_losses
non_trainable_variables
2regularization_losses
3trainable_variables
layer_metrics
layers
metrics
4	variables

VARIABLE_VALUE,seq_self_attention/seq_self_attention_Add_WtIlayer_with_weights-4/seq_self_attention_Add_Wt/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,seq_self_attention/seq_self_attention_Add_WxIlayer_with_weights-4/seq_self_attention_Add_Wx/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,seq_self_attention/seq_self_attention_Add_bhIlayer_with_weights-4/seq_self_attention_Add_bh/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,seq_self_attention/seq_self_attention_Add_WaIlayer_with_weights-4/seq_self_attention_Add_Wa/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,seq_self_attention/seq_self_attention_Add_baIlayer_with_weights-4/seq_self_attention_Add_ba/.ATTRIBUTES/VARIABLE_VALUE
#
60
71
82
93
:4
 
#
60
71
82
93
:4
ฒ
 layer_regularization_losses
non_trainable_variables
;trainable_variables
<regularization_losses
layer_metrics
layers
metrics
=	variables
l

Vkernel
Wbias
trainable_variables
regularization_losses
	variables
	keras_api

V0
W1
 

V0
W1
ฒ
 layer_regularization_losses
non_trainable_variables
@trainable_variables
Aregularization_losses
layer_metrics
layers
metrics
B	variables
l

Xkernel
Ybias
trainable_variables
regularization_losses
	variables
	keras_api

X0
Y1
 

X0
Y1
ฒ
 layer_regularization_losses
non_trainable_variables
Etrainable_variables
Fregularization_losses
layer_metrics
layers
metrics
G	variables
l

Zkernel
[bias
?trainable_variables
กregularization_losses
ข	variables
ฃ	keras_api

Z0
[1
 

Z0
[1
ฒ
 คlayer_regularization_losses
ฅnon_trainable_variables
Jtrainable_variables
Kregularization_losses
ฆlayer_metrics
งlayers
จmetrics
L	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUElstm/lstm_cell/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEtime_distributed/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEtime_distributed/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEtime_distributed_1/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEtime_distributed_1/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEtime_distributed_2/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEtime_distributed_2/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
N
0
1
2
3
4
5
6
7
	8

9
10

ฉ0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

S0
T1
U2
 

S0
T1
U2
ด
 ชlayer_regularization_losses
ซnon_trainable_variables
trainable_variables
regularization_losses
ฌlayer_metrics
ญlayers
ฎmetrics
	variables
 
 
 
 

00
 
 
 
 
 
 

V0
W1
 

V0
W1
ต
 ฏlayer_regularization_losses
ฐnon_trainable_variables
trainable_variables
regularization_losses
ฑlayer_metrics
ฒlayers
ณmetrics
	variables
 
 
 

?0
 

X0
Y1
 

X0
Y1
ต
 ดlayer_regularization_losses
ตnon_trainable_variables
trainable_variables
regularization_losses
ถlayer_metrics
ทlayers
ธmetrics
	variables
 
 
 

D0
 

Z0
[1
 

Z0
[1
ต
 นlayer_regularization_losses
บnon_trainable_variables
?trainable_variables
กregularization_losses
ปlayer_metrics
ผlayers
ฝmetrics
ข	variables
 
 
 

I0
 
8

พtotal

ฟcount
ภ	variables
ม	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

พ0
ฟ1

ภ	variables
|z
VARIABLE_VALUEAdam/conv1d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_Wt/melayer_with_weights-4/seq_self_attention_Add_Wt/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_Wx/melayer_with_weights-4/seq_self_attention_Add_Wx/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_bh/melayer_with_weights-4/seq_self_attention_Add_bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_Wa/melayer_with_weights-4/seq_self_attention_Add_Wa/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_ba/melayer_with_weights-4/seq_self_attention_Add_ba/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/time_distributed/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/time_distributed/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/time_distributed_1/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/time_distributed_1/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/time_distributed_2/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/time_distributed_2/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_Wt/velayer_with_weights-4/seq_self_attention_Add_Wt/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_Wx/velayer_with_weights-4/seq_self_attention_Add_Wx/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_bh/velayer_with_weights-4/seq_self_attention_Add_bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_Wa/velayer_with_weights-4/seq_self_attention_Add_Wa/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ฏฌ
VARIABLE_VALUE3Adam/seq_self_attention/seq_self_attention_Add_ba/velayer_with_weights-4/seq_self_attention_Add_ba/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/time_distributed/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/time_distributed/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/time_distributed_1/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/time_distributed_1/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/time_distributed_2/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/time_distributed_2/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv1d_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_inputconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias,seq_self_attention/seq_self_attention_Add_Wt,seq_self_attention/seq_self_attention_Add_Wx,seq_self_attention/seq_self_attention_Add_bh,seq_self_attention/seq_self_attention_Add_Wa,seq_self_attention/seq_self_attention_Add_batime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biastime_distributed_2/kerneltime_distributed_2/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_6188
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
๘
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp@seq_self_attention/seq_self_attention_Add_Wt/Read/ReadVariableOp@seq_self_attention/seq_self_attention_Add_Wx/Read/ReadVariableOp@seq_self_attention/seq_self_attention_Add_bh/Read/ReadVariableOp@seq_self_attention/seq_self_attention_Add_Wa/Read/ReadVariableOp@seq_self_attention/seq_self_attention_Add_ba/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOp-time_distributed_1/kernel/Read/ReadVariableOp+time_distributed_1/bias/Read/ReadVariableOp-time_distributed_2/kernel/Read/ReadVariableOp+time_distributed_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_Wt/m/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_Wx/m/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_bh/m/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_Wa/m/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_ba/m/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOp2Adam/time_distributed/kernel/m/Read/ReadVariableOp0Adam/time_distributed/bias/m/Read/ReadVariableOp4Adam/time_distributed_1/kernel/m/Read/ReadVariableOp2Adam/time_distributed_1/bias/m/Read/ReadVariableOp4Adam/time_distributed_2/kernel/m/Read/ReadVariableOp2Adam/time_distributed_2/bias/m/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_Wt/v/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_Wx/v/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_bh/v/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_Wa/v/Read/ReadVariableOpGAdam/seq_self_attention/seq_self_attention_Add_ba/v/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOp2Adam/time_distributed/kernel/v/Read/ReadVariableOp0Adam/time_distributed/bias/v/Read/ReadVariableOp4Adam/time_distributed_1/kernel/v/Read/ReadVariableOp2Adam/time_distributed_1/bias/v/Read/ReadVariableOp4Adam/time_distributed_2/kernel/v/Read/ReadVariableOp2Adam/time_distributed_2/bias/v/Read/ReadVariableOpConst*P
TinI
G2E	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_8553
ท
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/bias,seq_self_attention/seq_self_attention_Add_Wt,seq_self_attention/seq_self_attention_Add_Wx,seq_self_attention/seq_self_attention_Add_bh,seq_self_attention/seq_self_attention_Add_Wa,seq_self_attention/seq_self_attention_Add_ba	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biastime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biastime_distributed_2/kerneltime_distributed_2/biastotalcountAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/conv1d_2/kernel/mAdam/conv1d_2/bias/m3Adam/seq_self_attention/seq_self_attention_Add_Wt/m3Adam/seq_self_attention/seq_self_attention_Add_Wx/m3Adam/seq_self_attention/seq_self_attention_Add_bh/m3Adam/seq_self_attention/seq_self_attention_Add_Wa/m3Adam/seq_self_attention/seq_self_attention_Add_ba/mAdam/lstm/lstm_cell/kernel/m&Adam/lstm/lstm_cell/recurrent_kernel/mAdam/lstm/lstm_cell/bias/mAdam/time_distributed/kernel/mAdam/time_distributed/bias/m Adam/time_distributed_1/kernel/mAdam/time_distributed_1/bias/m Adam/time_distributed_2/kernel/mAdam/time_distributed_2/bias/mAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/v3Adam/seq_self_attention/seq_self_attention_Add_Wt/v3Adam/seq_self_attention/seq_self_attention_Add_Wx/v3Adam/seq_self_attention/seq_self_attention_Add_bh/v3Adam/seq_self_attention/seq_self_attention_Add_Wa/v3Adam/seq_self_attention/seq_self_attention_Add_ba/vAdam/lstm/lstm_cell/kernel/v&Adam/lstm/lstm_cell/recurrent_kernel/vAdam/lstm/lstm_cell/bias/vAdam/time_distributed/kernel/vAdam/time_distributed/bias/v Adam/time_distributed_1/kernel/vAdam/time_distributed_1/bias/v Adam/time_distributed_2/kernel/vAdam/time_distributed_2/bias/v*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_8764?ฺ!
ฏ
ฉ
A__inference_dense_1_layer_call_and_return_conditional_losses_8301

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๎U
๒
>__inference_lstm_layer_call_and_return_conditional_losses_7567

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityขwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shapeฒ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2ฟ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape๘
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2ฌ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02!
lstm_cell/MatMul/ReadVariableOpฃ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMulฑ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/addช
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim็
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_1
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shapeธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterแ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_7482*
condR
while_cond_7481*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
whileต
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape่
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permฅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::2
whilewhile:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ฑZ
?
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_5740

inputs#
shape_3_readvariableop_resource#
shape_5_readvariableop_resource!
add_1_readvariableop_resource#
shape_7_readvariableop_resource!
add_2_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2๎
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2H
Shape_2Shapeinputs*
T0*
_output_shapes
:2	
Shape_2^
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"
       2	
Shape_3`
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????
2	
Reshape
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:
 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:
 2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2ข
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_2b
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsReshape_2:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2

ExpandDimsH
Shape_4Shapeinputs*
T0*
_output_shapes
:2	
Shape_4b
	unstack_2UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02
Shape_5/ReadVariableOpc
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"
       2	
Shape_5`
	unstack_3UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape_3/shapeu
	Reshape_3ReshapeinputsReshape_3/shape:output:0*
T0*'
_output_shapes
:?????????
2
	Reshape_3
transpose_1/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:
 2
transpose_1s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:
 2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_5/shape/2ค
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_5f
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsReshape_5:output:0ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:????????? 2
ExpandDims_1y
addAddV2ExpandDims:output:0ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? 2
add
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
add_1/ReadVariableOpx
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
add_1Y
TanhTanh	add_1:z:0*
T0*/
_output_shapes
:????????? 2
TanhJ
Shape_6ShapeTanh:y:0*
T0*
_output_shapes
:2	
Shape_6d
	unstack_4UnpackShape_6:output:0*
T0*
_output_shapes

: : : : *	
num2
	unstack_4
Shape_7/ReadVariableOpReadVariableOpshape_7_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_7/ReadVariableOpc
Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_7`
	unstack_5UnpackShape_7:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_5s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape_6/shapew
	Reshape_6ReshapeTanh:y:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:????????? 2
	Reshape_6
transpose_2/ReadVariableOpReadVariableOpshape_7_readvariableop_resource*
_output_shapes

: *
dtype02
transpose_2/ReadVariableOpu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_7/shapeu
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_7x
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2h
Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/1h
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/2h
Reshape_8/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/3ภ
Reshape_8/shapePackunstack_4:output:0Reshape_8/shape/1:output:0Reshape_8/shape/2:output:0Reshape_8/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_8/shape
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*/
_output_shapes
:?????????2
	Reshape_8
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
add_2/ReadVariableOp
add_2AddV2Reshape_8:output:0add_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
add_2ฆ
Reshape_9/shapePackstrided_slice_1:output:0strided_slice_2:output:0strided_slice_2:output:0*
N*
T0*
_output_shapes
:2
Reshape_9/shape
	Reshape_9Reshape	add_2:z:0Reshape_9/shape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Reshape_9p
ReluReluReshape_9:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
Reluy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices
MaxMaxRelu:activations:0Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Max{
subSubRelu:activations:0Max:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
subb
ExpExpsub:z:0*
T0*=
_output_shapes+
):'???????????????????????????2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Sum|
truedivRealDivExp:y:0Sum:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
truedivy
MatMul_3BatchMatMulV2truediv:z:0inputs*
T0*4
_output_shapes"
 :??????????????????
2

MatMul_3S
Shape_8ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_8r
IdentityIdentityMatMul_3:output:0*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
ญ

#__inference_lstm_layer_call_fn_7403
inputs_0
unknown
	unknown_0
	unknown_1
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_45692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:??????????????????
"
_user_specified_name
inputs/0
๑
H
,__inference_max_pooling1d_layer_call_fn_4085

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_40792
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

ฤ
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_8152

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:?????????2	
Reshapeฆ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMulค
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpก
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1s
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????:::] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ๆ
z
%__inference_conv1d_layer_call_fn_7025

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_50952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
อ
ฉ
A__inference_dense_2_layer_call_and_return_conditional_losses_4975

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ก
น
while_cond_7306
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_7306___redundant_placeholder02
.while_while_cond_7306___redundant_placeholder12
.while_while_cond_7306___redundant_placeholder22
.while_while_cond_7306___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:


#__inference_lstm_layer_call_fn_7742

inputs
unknown
	unknown_0
	unknown_1
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_54962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ทํ
ษ	
D__inference_sequential_layer_call_and_return_conditional_losses_6910

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource6
2seq_self_attention_shape_3_readvariableop_resource6
2seq_self_attention_shape_5_readvariableop_resource4
0seq_self_attention_add_1_readvariableop_resource6
2seq_self_attention_shape_7_readvariableop_resource4
0seq_self_attention_add_2_readvariableop_resource9
5time_distributed_dense_matmul_readvariableop_resource:
6time_distributed_dense_biasadd_readvariableop_resource=
9time_distributed_1_dense_1_matmul_readvariableop_resource>
:time_distributed_1_dense_1_biasadd_readvariableop_resource=
9time_distributed_2_dense_2_matmul_readvariableop_resource>
:time_distributed_2_dense_2_biasadd_readvariableop_resource
identityข
lstm/while
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dimซ
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/conv1d/ExpandDimsฮ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimิ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:2
conv1d/conv1d/ExpandDims_1ิ
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1dจ
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeezeข
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1d/BiasAdd/ReadVariableOpฉ
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2
conv1d/BiasAddr
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????2
conv1d/Relu
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dimล
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d_1/conv1d/ExpandDimsี
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_1/conv1dฎ
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeezeจ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOpฑ
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2
conv1d_1/BiasAddx
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:?????????2
conv1d_1/Relu
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dimว
conv1d_2/conv1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d_2/conv1d/ExpandDimsี
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_2/conv1dฎ
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeezeจ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOpฑ
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2
conv1d_2/BiasAddx
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????2
conv1d_2/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dimม
max_pooling1d/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
max_pooling1d/ExpandDimsส
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPoolง
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims
2
max_pooling1d/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const
flatten/ReshapeReshapemax_pooling1d/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten/Reshape~
repeat_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
repeat_vector/ExpandDims/dimบ
repeat_vector/ExpandDims
ExpandDimsflatten/Reshape:output:0%repeat_vector/ExpandDims/dim:output:0*
T0*,
_output_shapes
:?????????2
repeat_vector/ExpandDims
repeat_vector/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
repeat_vector/stackจ
repeat_vector/TileTile!repeat_vector/ExpandDims:output:0repeat_vector/stack:output:0*
T0*,
_output_shapes
:?????????2
repeat_vector/Tilec

lstm/ShapeShaperepeat_vector/Tile:output:0*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm
lstm/transpose	Transposerepeat_vector/Tile:output:0lstm/transpose/perm:output:0*
T0*,
_output_shapes
:?????????2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm/TensorArrayV2/element_shapeฦ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2ษ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
lstm/strided_slice_2ป
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOpท
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/lstm_cell/MatMulภ
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpณ
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/lstm_cell/MatMul_1ง
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm/lstm_cell/addน
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOpด
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim๛
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm/lstm_cell/split
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Sigmoid
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Sigmoid_1
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/mul
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Reluค
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/mul_1
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/add_1
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Sigmoid_2
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Relu_1จ
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/mul_2
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2$
"lstm/TensorArrayV2_1/element_shapeฬ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterฌ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
lstm_while_body_6663* 
condR
lstm_while_cond_6662*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2

lstm/whileฟ
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2ธ
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permน
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimex
seq_self_attention/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape
&seq_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&seq_self_attention/strided_slice/stack
(seq_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(seq_self_attention/strided_slice/stack_1
(seq_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(seq_self_attention/strided_slice/stack_2ิ
 seq_self_attention/strided_sliceStridedSlice!seq_self_attention/Shape:output:0/seq_self_attention/strided_slice/stack:output:01seq_self_attention/strided_slice/stack_1:output:01seq_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 seq_self_attention/strided_slice|
seq_self_attention/Shape_1Shapelstm/transpose_1:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_1
(seq_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(seq_self_attention/strided_slice_1/stackข
*seq_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*seq_self_attention/strided_slice_1/stack_1ข
*seq_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*seq_self_attention/strided_slice_1/stack_2เ
"seq_self_attention/strided_slice_1StridedSlice#seq_self_attention/Shape_1:output:01seq_self_attention/strided_slice_1/stack:output:03seq_self_attention/strided_slice_1/stack_1:output:03seq_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"seq_self_attention/strided_slice_1
(seq_self_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(seq_self_attention/strided_slice_2/stackข
*seq_self_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*seq_self_attention/strided_slice_2/stack_1ข
*seq_self_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*seq_self_attention/strided_slice_2/stack_2เ
"seq_self_attention/strided_slice_2StridedSlice#seq_self_attention/Shape_1:output:01seq_self_attention/strided_slice_2/stack:output:03seq_self_attention/strided_slice_2/stack_1:output:03seq_self_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"seq_self_attention/strided_slice_2|
seq_self_attention/Shape_2Shapelstm/transpose_1:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_2
seq_self_attention/unstackUnpack#seq_self_attention/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
seq_self_attention/unstackษ
)seq_self_attention/Shape_3/ReadVariableOpReadVariableOp2seq_self_attention_shape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02+
)seq_self_attention/Shape_3/ReadVariableOp
seq_self_attention/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"
       2
seq_self_attention/Shape_3
seq_self_attention/unstack_1Unpack#seq_self_attention/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
seq_self_attention/unstack_1
 seq_self_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2"
 seq_self_attention/Reshape/shapeถ
seq_self_attention/ReshapeReshapelstm/transpose_1:y:0)seq_self_attention/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2
seq_self_attention/Reshapeอ
+seq_self_attention/transpose/ReadVariableOpReadVariableOp2seq_self_attention_shape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02-
+seq_self_attention/transpose/ReadVariableOp
!seq_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2#
!seq_self_attention/transpose/permำ
seq_self_attention/transpose	Transpose3seq_self_attention/transpose/ReadVariableOp:value:0*seq_self_attention/transpose/perm:output:0*
T0*
_output_shapes

:
 2
seq_self_attention/transpose
"seq_self_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2$
"seq_self_attention/Reshape_1/shapeฟ
seq_self_attention/Reshape_1Reshape seq_self_attention/transpose:y:0+seq_self_attention/Reshape_1/shape:output:0*
T0*
_output_shapes

:
 2
seq_self_attention/Reshape_1พ
seq_self_attention/MatMulMatMul#seq_self_attention/Reshape:output:0%seq_self_attention/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
seq_self_attention/MatMul
$seq_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_2/shape/1
$seq_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2&
$seq_self_attention/Reshape_2/shape/2
"seq_self_attention/Reshape_2/shapePack#seq_self_attention/unstack:output:0-seq_self_attention/Reshape_2/shape/1:output:0-seq_self_attention/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"seq_self_attention/Reshape_2/shapeฯ
seq_self_attention/Reshape_2Reshape#seq_self_attention/MatMul:product:0+seq_self_attention/Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
seq_self_attention/Reshape_2
!seq_self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!seq_self_attention/ExpandDims/dimู
seq_self_attention/ExpandDims
ExpandDims%seq_self_attention/Reshape_2:output:0*seq_self_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
seq_self_attention/ExpandDims|
seq_self_attention/Shape_4Shapelstm/transpose_1:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_4
seq_self_attention/unstack_2Unpack#seq_self_attention/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num2
seq_self_attention/unstack_2ษ
)seq_self_attention/Shape_5/ReadVariableOpReadVariableOp2seq_self_attention_shape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02+
)seq_self_attention/Shape_5/ReadVariableOp
seq_self_attention/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"
       2
seq_self_attention/Shape_5
seq_self_attention/unstack_3Unpack#seq_self_attention/Shape_5:output:0*
T0*
_output_shapes
: : *	
num2
seq_self_attention/unstack_3
"seq_self_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2$
"seq_self_attention/Reshape_3/shapeผ
seq_self_attention/Reshape_3Reshapelstm/transpose_1:y:0+seq_self_attention/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????
2
seq_self_attention/Reshape_3ั
-seq_self_attention/transpose_1/ReadVariableOpReadVariableOp2seq_self_attention_shape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02/
-seq_self_attention/transpose_1/ReadVariableOp
#seq_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2%
#seq_self_attention/transpose_1/perm?
seq_self_attention/transpose_1	Transpose5seq_self_attention/transpose_1/ReadVariableOp:value:0,seq_self_attention/transpose_1/perm:output:0*
T0*
_output_shapes

:
 2 
seq_self_attention/transpose_1
"seq_self_attention/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2$
"seq_self_attention/Reshape_4/shapeม
seq_self_attention/Reshape_4Reshape"seq_self_attention/transpose_1:y:0+seq_self_attention/Reshape_4/shape:output:0*
T0*
_output_shapes

:
 2
seq_self_attention/Reshape_4ฤ
seq_self_attention/MatMul_1MatMul%seq_self_attention/Reshape_3:output:0%seq_self_attention/Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2
seq_self_attention/MatMul_1
$seq_self_attention/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_5/shape/1
$seq_self_attention/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2&
$seq_self_attention/Reshape_5/shape/2
"seq_self_attention/Reshape_5/shapePack%seq_self_attention/unstack_2:output:0-seq_self_attention/Reshape_5/shape/1:output:0-seq_self_attention/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"seq_self_attention/Reshape_5/shapeั
seq_self_attention/Reshape_5Reshape%seq_self_attention/MatMul_1:product:0+seq_self_attention/Reshape_5/shape:output:0*
T0*+
_output_shapes
:????????? 2
seq_self_attention/Reshape_5
#seq_self_attention/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#seq_self_attention/ExpandDims_1/dim฿
seq_self_attention/ExpandDims_1
ExpandDims%seq_self_attention/Reshape_5:output:0,seq_self_attention/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:????????? 2!
seq_self_attention/ExpandDims_1ล
seq_self_attention/addAddV2&seq_self_attention/ExpandDims:output:0(seq_self_attention/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? 2
seq_self_attention/addฟ
'seq_self_attention/add_1/ReadVariableOpReadVariableOp0seq_self_attention_add_1_readvariableop_resource*
_output_shapes
: *
dtype02)
'seq_self_attention/add_1/ReadVariableOpฤ
seq_self_attention/add_1AddV2seq_self_attention/add:z:0/seq_self_attention/add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
seq_self_attention/add_1
seq_self_attention/TanhTanhseq_self_attention/add_1:z:0*
T0*/
_output_shapes
:????????? 2
seq_self_attention/Tanh
seq_self_attention/Shape_6Shapeseq_self_attention/Tanh:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_6
seq_self_attention/unstack_4Unpack#seq_self_attention/Shape_6:output:0*
T0*
_output_shapes

: : : : *	
num2
seq_self_attention/unstack_4ษ
)seq_self_attention/Shape_7/ReadVariableOpReadVariableOp2seq_self_attention_shape_7_readvariableop_resource*
_output_shapes

: *
dtype02+
)seq_self_attention/Shape_7/ReadVariableOp
seq_self_attention/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2
seq_self_attention/Shape_7
seq_self_attention/unstack_5Unpack#seq_self_attention/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2
seq_self_attention/unstack_5
"seq_self_attention/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2$
"seq_self_attention/Reshape_6/shapeร
seq_self_attention/Reshape_6Reshapeseq_self_attention/Tanh:y:0+seq_self_attention/Reshape_6/shape:output:0*
T0*'
_output_shapes
:????????? 2
seq_self_attention/Reshape_6ั
-seq_self_attention/transpose_2/ReadVariableOpReadVariableOp2seq_self_attention_shape_7_readvariableop_resource*
_output_shapes

: *
dtype02/
-seq_self_attention/transpose_2/ReadVariableOp
#seq_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2%
#seq_self_attention/transpose_2/perm?
seq_self_attention/transpose_2	Transpose5seq_self_attention/transpose_2/ReadVariableOp:value:0,seq_self_attention/transpose_2/perm:output:0*
T0*
_output_shapes

: 2 
seq_self_attention/transpose_2
"seq_self_attention/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2$
"seq_self_attention/Reshape_7/shapeม
seq_self_attention/Reshape_7Reshape"seq_self_attention/transpose_2:y:0+seq_self_attention/Reshape_7/shape:output:0*
T0*
_output_shapes

: 2
seq_self_attention/Reshape_7ฤ
seq_self_attention/MatMul_2MatMul%seq_self_attention/Reshape_6:output:0%seq_self_attention/Reshape_7:output:0*
T0*'
_output_shapes
:?????????2
seq_self_attention/MatMul_2
$seq_self_attention/Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_8/shape/1
$seq_self_attention/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_8/shape/2
$seq_self_attention/Reshape_8/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_8/shape/3ฒ
"seq_self_attention/Reshape_8/shapePack%seq_self_attention/unstack_4:output:0-seq_self_attention/Reshape_8/shape/1:output:0-seq_self_attention/Reshape_8/shape/2:output:0-seq_self_attention/Reshape_8/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"seq_self_attention/Reshape_8/shapeี
seq_self_attention/Reshape_8Reshape%seq_self_attention/MatMul_2:product:0+seq_self_attention/Reshape_8/shape:output:0*
T0*/
_output_shapes
:?????????2
seq_self_attention/Reshape_8ฟ
'seq_self_attention/add_2/ReadVariableOpReadVariableOp0seq_self_attention_add_2_readvariableop_resource*
_output_shapes
:*
dtype02)
'seq_self_attention/add_2/ReadVariableOpฯ
seq_self_attention/add_2AddV2%seq_self_attention/Reshape_8:output:0/seq_self_attention/add_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
seq_self_attention/add_2
"seq_self_attention/Reshape_9/shapePack+seq_self_attention/strided_slice_1:output:0+seq_self_attention/strided_slice_2:output:0+seq_self_attention/strided_slice_2:output:0*
N*
T0*
_output_shapes
:2$
"seq_self_attention/Reshape_9/shapeฺ
seq_self_attention/Reshape_9Reshapeseq_self_attention/add_2:z:0+seq_self_attention/Reshape_9/shape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/Reshape_9ฉ
seq_self_attention/ReluRelu%seq_self_attention/Reshape_9:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/Relu
(seq_self_attention/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(seq_self_attention/Max/reduction_indicesแ
seq_self_attention/MaxMax%seq_self_attention/Relu:activations:01seq_self_attention/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
seq_self_attention/Maxว
seq_self_attention/subSub%seq_self_attention/Relu:activations:0seq_self_attention/Max:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/sub
seq_self_attention/ExpExpseq_self_attention/sub:z:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/Exp
(seq_self_attention/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(seq_self_attention/Sum/reduction_indicesึ
seq_self_attention/SumSumseq_self_attention/Exp:y:01seq_self_attention/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
seq_self_attention/Sumศ
seq_self_attention/truedivRealDivseq_self_attention/Exp:y:0seq_self_attention/Sum:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/truedivภ
seq_self_attention/MatMul_3BatchMatMulV2seq_self_attention/truediv:z:0lstm/transpose_1:y:0*
T0*4
_output_shapes"
 :??????????????????
2
seq_self_attention/MatMul_3
seq_self_attention/Shape_8Shape$seq_self_attention/MatMul_3:output:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_8
time_distributed/ShapeShape$seq_self_attention/MatMul_3:output:0*
T0*
_output_shapes
:2
time_distributed/Shape
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$time_distributed/strided_slice/stack
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed/strided_slice/stack_1
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed/strided_slice/stack_2ศ
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
time_distributed/strided_slice
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2 
time_distributed/Reshape/shapeภ
time_distributed/ReshapeReshape$seq_self_attention/MatMul_3:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2
time_distributed/Reshapeำ
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOp5time_distributed_dense_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02.
,time_distributed/dense/MatMul/ReadVariableOpิ
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
time_distributed/dense/MatMulา
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOp6time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-time_distributed/dense/BiasAdd/ReadVariableOp?
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2 
time_distributed/dense/BiasAdd
time_distributed/dense/ReluRelu'time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
time_distributed/dense/Relu
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"time_distributed/Reshape_1/shape/0
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2$
"time_distributed/Reshape_1/shape/2?
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 time_distributed/Reshape_1/shapeู
time_distributed/Reshape_1Reshape)time_distributed/dense/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
time_distributed/Reshape_1
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2"
 time_distributed/Reshape_2/shapeฦ
time_distributed/Reshape_2Reshape$seq_self_attention/MatMul_3:output:0)time_distributed/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????
2
time_distributed/Reshape_2
time_distributed_1/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_1/Shape
&time_distributed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_1/strided_slice/stack
(time_distributed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_1/strided_slice/stack_1
(time_distributed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_1/strided_slice/stack_2ิ
 time_distributed_1/strided_sliceStridedSlice!time_distributed_1/Shape:output:0/time_distributed_1/strided_slice/stack:output:01time_distributed_1/strided_slice/stack_1:output:01time_distributed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_1/strided_slice
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_1/Reshape/shapeฦ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_1/Reshapeเ
0time_distributed_1/dense_1/MatMul/ReadVariableOpReadVariableOp9time_distributed_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0time_distributed_1/dense_1/MatMul/ReadVariableOpโ
!time_distributed_1/dense_1/MatMulMatMul#time_distributed_1/Reshape:output:08time_distributed_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2#
!time_distributed_1/dense_1/MatMul?
1time_distributed_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1time_distributed_1/dense_1/BiasAdd/ReadVariableOp๎
"time_distributed_1/dense_1/BiasAddBiasAdd+time_distributed_1/dense_1/MatMul:product:09time_distributed_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2$
"time_distributed_1/dense_1/BiasAddช
time_distributed_1/dense_1/ReluRelu+time_distributed_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2!
time_distributed_1/dense_1/Relu
$time_distributed_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_1/Reshape_1/shape/0
$time_distributed_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2&
$time_distributed_1/Reshape_1/shape/2
"time_distributed_1/Reshape_1/shapePack-time_distributed_1/Reshape_1/shape/0:output:0)time_distributed_1/strided_slice:output:0-time_distributed_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_1/Reshape_1/shapeใ
time_distributed_1/Reshape_1Reshape-time_distributed_1/dense_1/Relu:activations:0+time_distributed_1/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
time_distributed_1/Reshape_1
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2$
"time_distributed_1/Reshape_2/shapeฬ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_1/Reshape_2
time_distributed_2/ShapeShape%time_distributed_1/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_2/Shape
&time_distributed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_2/strided_slice/stack
(time_distributed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_2/strided_slice/stack_1
(time_distributed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_2/strided_slice/stack_2ิ
 time_distributed_2/strided_sliceStridedSlice!time_distributed_2/Shape:output:0/time_distributed_2/strided_slice/stack:output:01time_distributed_2/strided_slice/stack_1:output:01time_distributed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_2/strided_slice
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_2/Reshape/shapeศ
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_2/Reshape฿
0time_distributed_2/dense_2/MatMul/ReadVariableOpReadVariableOp9time_distributed_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0time_distributed_2/dense_2/MatMul/ReadVariableOpแ
!time_distributed_2/dense_2/MatMulMatMul#time_distributed_2/Reshape:output:08time_distributed_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!time_distributed_2/dense_2/MatMul?
1time_distributed_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1time_distributed_2/dense_2/BiasAdd/ReadVariableOpํ
"time_distributed_2/dense_2/BiasAddBiasAdd+time_distributed_2/dense_2/MatMul:product:09time_distributed_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"time_distributed_2/dense_2/BiasAdd
$time_distributed_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_2/Reshape_1/shape/0
$time_distributed_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$time_distributed_2/Reshape_1/shape/2
"time_distributed_2/Reshape_1/shapePack-time_distributed_2/Reshape_1/shape/0:output:0)time_distributed_2/strided_slice:output:0-time_distributed_2/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_2/Reshape_1/shapeเ
time_distributed_2/Reshape_1Reshape+time_distributed_2/dense_2/BiasAdd:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
time_distributed_2/Reshape_1
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2$
"time_distributed_2/Reshape_2/shapeฮ
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_2/Reshape_2
IdentityIdentity%time_distributed_2/Reshape_1:output:0^lstm/while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::2

lstm/while
lstm/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
๋
พ
J__inference_time_distributed_layer_call_and_return_conditional_losses_8008

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????
2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2

dense/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapedense/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????
:::\ X
4
_output_shapes"
 :??????????????????

 
_user_specified_nameinputs
ณ
ศ
(__inference_lstm_cell_layer_call_fn_8253

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ขStatefulPartitionedCallพ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????
:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_41732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:?????????:?????????
:?????????
:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/1


/__inference_time_distributed_layer_call_fn_8039

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_47962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????
::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????

 
_user_specified_nameinputs
D
ล
>__inference_lstm_layer_call_and_return_conditional_losses_4569

inputs
lstm_cell_4487
lstm_cell_4489
lstm_cell_4491
identityข!lstm_cell/StatefulPartitionedCallขwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shapeฒ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2ฟ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape๘
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4487lstm_cell_4489lstm_cell_4491*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????
:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_41732#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shapeธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4487lstm_cell_4489lstm_cell_4491*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4500*
condR
while_cond_4499*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
whileต
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape๑
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permฎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ใ

)__inference_sequential_layer_call_fn_6133
conv1d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityขStatefulPartitionedCall๚
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_60902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
๒7
ๆ
while_body_5411
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceร
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeิ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemภ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	(*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpอ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMulล
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpถ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMul_1ซ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/addพ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpธ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Reluจ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Relu_1ฌ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 

ฑ
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_5068

inputs
dense_2_5058
dense_2_5060
identityขdense_2/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:?????????2	
Reshape
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_2_5058dense_2_5060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_49752!
dense_2/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeค
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1
IdentityIdentityReshape_1:output:0 ^dense_2/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs

B
&__inference_flatten_layer_call_fn_7086

inputs
identityภ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_51822
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
๒7
ๆ
while_body_7307
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceร
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeิ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemภ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	(*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpอ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMulล
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpถ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMul_1ซ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/addพ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpธ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Reluจ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Relu_1ฌ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
ย@
๑
D__inference_sequential_layer_call_and_return_conditional_losses_6090

inputs
conv1d_6032
conv1d_6034
conv1d_1_6037
conv1d_1_6039
conv1d_2_6042
conv1d_2_6044
	lstm_6050
	lstm_6052
	lstm_6054
seq_self_attention_6057
seq_self_attention_6059
seq_self_attention_6061
seq_self_attention_6063
seq_self_attention_6065
time_distributed_6068
time_distributed_6070
time_distributed_1_6075
time_distributed_1_6077
time_distributed_2_6082
time_distributed_2_6084
identityขconv1d/StatefulPartitionedCallข conv1d_1/StatefulPartitionedCallข conv1d_2/StatefulPartitionedCallขlstm/StatefulPartitionedCallข*seq_self_attention/StatefulPartitionedCallข(time_distributed/StatefulPartitionedCallข*time_distributed_1/StatefulPartitionedCallข*time_distributed_2/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_6032conv1d_6034*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_50952 
conv1d/StatefulPartitionedCallด
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_6037conv1d_1_6039*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_51272"
 conv1d_1/StatefulPartitionedCallถ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_6042conv1d_2_6044*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_51592"
 conv1d_2/StatefulPartitionedCall
max_pooling1d/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_40792
max_pooling1d/PartitionedCall๐
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_51822
flatten/PartitionedCall
repeat_vector/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_repeat_vector_layer_call_and_return_conditional_losses_40942
repeat_vector/PartitionedCallซ
lstm/StatefulPartitionedCallStatefulPartitionedCall&repeat_vector/PartitionedCall:output:0	lstm_6050	lstm_6052	lstm_6054*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_54962
lstm/StatefulPartitionedCallฝ
*seq_self_attention/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0seq_self_attention_6057seq_self_attention_6059seq_self_attention_6061seq_self_attention_6063seq_self_attention_6065*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_57402,
*seq_self_attention/StatefulPartitionedCall๑
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall3seq_self_attention/StatefulPartitionedCall:output:0time_distributed_6068time_distributed_6070*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_48252*
(time_distributed/StatefulPartitionedCall
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2 
time_distributed/Reshape/shapeฯ
time_distributed/ReshapeReshape3seq_self_attention/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2
time_distributed/Reshape๙
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_6075time_distributed_1_6077*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_49472,
*time_distributed_1/StatefulPartitionedCall
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_1/Reshape/shapeิ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_1/Reshape๚
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0time_distributed_2_6082time_distributed_2_6084*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_50682,
*time_distributed_2/StatefulPartitionedCall
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_2/Reshape/shapeึ
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_2/Reshapeฬ
IdentityIdentity3time_distributed_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall+^seq_self_attention/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2X
*seq_self_attention/StatefulPartitionedCall*seq_self_attention/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs

ฑ
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_5039

inputs
dense_2_5029
dense_2_5031
identityขdense_2/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:?????????2	
Reshape
dense_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_2_5029dense_2_5031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_49752!
dense_2/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeค
	Reshape_1Reshape(dense_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1
IdentityIdentityReshape_1:output:0 ^dense_2/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ณ
]
A__inference_flatten_layer_call_and_return_conditional_losses_5182

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ฅ
(
 __inference__traced_restore_8764
file_prefix"
assignvariableop_conv1d_kernel"
assignvariableop_1_conv1d_bias&
"assignvariableop_2_conv1d_1_kernel$
 assignvariableop_3_conv1d_1_bias&
"assignvariableop_4_conv1d_2_kernel$
 assignvariableop_5_conv1d_2_biasC
?assignvariableop_6_seq_self_attention_seq_self_attention_add_wtC
?assignvariableop_7_seq_self_attention_seq_self_attention_add_wxC
?assignvariableop_8_seq_self_attention_seq_self_attention_add_bhC
?assignvariableop_9_seq_self_attention_seq_self_attention_add_waD
@assignvariableop_10_seq_self_attention_seq_self_attention_add_ba!
assignvariableop_11_adam_iter#
assignvariableop_12_adam_beta_1#
assignvariableop_13_adam_beta_2"
assignvariableop_14_adam_decay*
&assignvariableop_15_adam_learning_rate-
)assignvariableop_16_lstm_lstm_cell_kernel7
3assignvariableop_17_lstm_lstm_cell_recurrent_kernel+
'assignvariableop_18_lstm_lstm_cell_bias/
+assignvariableop_19_time_distributed_kernel-
)assignvariableop_20_time_distributed_bias1
-assignvariableop_21_time_distributed_1_kernel/
+assignvariableop_22_time_distributed_1_bias1
-assignvariableop_23_time_distributed_2_kernel/
+assignvariableop_24_time_distributed_2_bias
assignvariableop_25_total
assignvariableop_26_count,
(assignvariableop_27_adam_conv1d_kernel_m*
&assignvariableop_28_adam_conv1d_bias_m.
*assignvariableop_29_adam_conv1d_1_kernel_m,
(assignvariableop_30_adam_conv1d_1_bias_m.
*assignvariableop_31_adam_conv1d_2_kernel_m,
(assignvariableop_32_adam_conv1d_2_bias_mK
Gassignvariableop_33_adam_seq_self_attention_seq_self_attention_add_wt_mK
Gassignvariableop_34_adam_seq_self_attention_seq_self_attention_add_wx_mK
Gassignvariableop_35_adam_seq_self_attention_seq_self_attention_add_bh_mK
Gassignvariableop_36_adam_seq_self_attention_seq_self_attention_add_wa_mK
Gassignvariableop_37_adam_seq_self_attention_seq_self_attention_add_ba_m4
0assignvariableop_38_adam_lstm_lstm_cell_kernel_m>
:assignvariableop_39_adam_lstm_lstm_cell_recurrent_kernel_m2
.assignvariableop_40_adam_lstm_lstm_cell_bias_m6
2assignvariableop_41_adam_time_distributed_kernel_m4
0assignvariableop_42_adam_time_distributed_bias_m8
4assignvariableop_43_adam_time_distributed_1_kernel_m6
2assignvariableop_44_adam_time_distributed_1_bias_m8
4assignvariableop_45_adam_time_distributed_2_kernel_m6
2assignvariableop_46_adam_time_distributed_2_bias_m,
(assignvariableop_47_adam_conv1d_kernel_v*
&assignvariableop_48_adam_conv1d_bias_v.
*assignvariableop_49_adam_conv1d_1_kernel_v,
(assignvariableop_50_adam_conv1d_1_bias_v.
*assignvariableop_51_adam_conv1d_2_kernel_v,
(assignvariableop_52_adam_conv1d_2_bias_vK
Gassignvariableop_53_adam_seq_self_attention_seq_self_attention_add_wt_vK
Gassignvariableop_54_adam_seq_self_attention_seq_self_attention_add_wx_vK
Gassignvariableop_55_adam_seq_self_attention_seq_self_attention_add_bh_vK
Gassignvariableop_56_adam_seq_self_attention_seq_self_attention_add_wa_vK
Gassignvariableop_57_adam_seq_self_attention_seq_self_attention_add_ba_v4
0assignvariableop_58_adam_lstm_lstm_cell_kernel_v>
:assignvariableop_59_adam_lstm_lstm_cell_recurrent_kernel_v2
.assignvariableop_60_adam_lstm_lstm_cell_bias_v6
2assignvariableop_61_adam_time_distributed_kernel_v4
0assignvariableop_62_adam_time_distributed_bias_v8
4assignvariableop_63_adam_time_distributed_1_kernel_v6
2assignvariableop_64_adam_time_distributed_1_bias_v8
4assignvariableop_65_adam_time_distributed_2_kernel_v6
2assignvariableop_66_adam_time_distributed_2_bias_v
identity_68ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_17ขAssignVariableOp_18ขAssignVariableOp_19ขAssignVariableOp_2ขAssignVariableOp_20ขAssignVariableOp_21ขAssignVariableOp_22ขAssignVariableOp_23ขAssignVariableOp_24ขAssignVariableOp_25ขAssignVariableOp_26ขAssignVariableOp_27ขAssignVariableOp_28ขAssignVariableOp_29ขAssignVariableOp_3ขAssignVariableOp_30ขAssignVariableOp_31ขAssignVariableOp_32ขAssignVariableOp_33ขAssignVariableOp_34ขAssignVariableOp_35ขAssignVariableOp_36ขAssignVariableOp_37ขAssignVariableOp_38ขAssignVariableOp_39ขAssignVariableOp_4ขAssignVariableOp_40ขAssignVariableOp_41ขAssignVariableOp_42ขAssignVariableOp_43ขAssignVariableOp_44ขAssignVariableOp_45ขAssignVariableOp_46ขAssignVariableOp_47ขAssignVariableOp_48ขAssignVariableOp_49ขAssignVariableOp_5ขAssignVariableOp_50ขAssignVariableOp_51ขAssignVariableOp_52ขAssignVariableOp_53ขAssignVariableOp_54ขAssignVariableOp_55ขAssignVariableOp_56ขAssignVariableOp_57ขAssignVariableOp_58ขAssignVariableOp_59ขAssignVariableOp_6ขAssignVariableOp_60ขAssignVariableOp_61ขAssignVariableOp_62ขAssignVariableOp_63ขAssignVariableOp_64ขAssignVariableOp_65ขAssignVariableOp_66ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9๓'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?&
value๕&B๒&DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_Wt/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_Wx/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_bh/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_Wa/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_ba/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wt/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wx/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wa/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_ba/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wt/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wx/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wa/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_ba/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ฆ
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ฃ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ง
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ฅ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ง
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ฅ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ฤ
AssignVariableOp_6AssignVariableOp?assignvariableop_6_seq_self_attention_seq_self_attention_add_wtIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ฤ
AssignVariableOp_7AssignVariableOp?assignvariableop_7_seq_self_attention_seq_self_attention_add_wxIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ฤ
AssignVariableOp_8AssignVariableOp?assignvariableop_8_seq_self_attention_seq_self_attention_add_bhIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ฤ
AssignVariableOp_9AssignVariableOp?assignvariableop_9_seq_self_attention_seq_self_attention_add_waIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ศ
AssignVariableOp_10AssignVariableOp@assignvariableop_10_seq_self_attention_seq_self_attention_add_baIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11ฅ
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ง
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ง
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ฆ
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ฎ
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ฑ
AssignVariableOp_16AssignVariableOp)assignvariableop_16_lstm_lstm_cell_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ป
AssignVariableOp_17AssignVariableOp3assignvariableop_17_lstm_lstm_cell_recurrent_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ฏ
AssignVariableOp_18AssignVariableOp'assignvariableop_18_lstm_lstm_cell_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ณ
AssignVariableOp_19AssignVariableOp+assignvariableop_19_time_distributed_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ฑ
AssignVariableOp_20AssignVariableOp)assignvariableop_20_time_distributed_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ต
AssignVariableOp_21AssignVariableOp-assignvariableop_21_time_distributed_1_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ณ
AssignVariableOp_22AssignVariableOp+assignvariableop_22_time_distributed_1_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ต
AssignVariableOp_23AssignVariableOp-assignvariableop_23_time_distributed_2_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ณ
AssignVariableOp_24AssignVariableOp+assignvariableop_24_time_distributed_2_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ก
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ก
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ฐ
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv1d_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ฎ
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv1d_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ฒ
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv1d_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ฐ
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv1d_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ฒ
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv1d_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ฐ
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv1d_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ฯ
AssignVariableOp_33AssignVariableOpGassignvariableop_33_adam_seq_self_attention_seq_self_attention_add_wt_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ฯ
AssignVariableOp_34AssignVariableOpGassignvariableop_34_adam_seq_self_attention_seq_self_attention_add_wx_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ฯ
AssignVariableOp_35AssignVariableOpGassignvariableop_35_adam_seq_self_attention_seq_self_attention_add_bh_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ฯ
AssignVariableOp_36AssignVariableOpGassignvariableop_36_adam_seq_self_attention_seq_self_attention_add_wa_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ฯ
AssignVariableOp_37AssignVariableOpGassignvariableop_37_adam_seq_self_attention_seq_self_attention_add_ba_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ธ
AssignVariableOp_38AssignVariableOp0assignvariableop_38_adam_lstm_lstm_cell_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39ย
AssignVariableOp_39AssignVariableOp:assignvariableop_39_adam_lstm_lstm_cell_recurrent_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ถ
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_lstm_lstm_cell_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41บ
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_time_distributed_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ธ
AssignVariableOp_42AssignVariableOp0assignvariableop_42_adam_time_distributed_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43ผ
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_time_distributed_1_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44บ
AssignVariableOp_44AssignVariableOp2assignvariableop_44_adam_time_distributed_1_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45ผ
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_time_distributed_2_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46บ
AssignVariableOp_46AssignVariableOp2assignvariableop_46_adam_time_distributed_2_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47ฐ
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_conv1d_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ฎ
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_conv1d_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49ฒ
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv1d_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50ฐ
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv1d_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51ฒ
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv1d_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52ฐ
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv1d_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53ฯ
AssignVariableOp_53AssignVariableOpGassignvariableop_53_adam_seq_self_attention_seq_self_attention_add_wt_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54ฯ
AssignVariableOp_54AssignVariableOpGassignvariableop_54_adam_seq_self_attention_seq_self_attention_add_wx_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55ฯ
AssignVariableOp_55AssignVariableOpGassignvariableop_55_adam_seq_self_attention_seq_self_attention_add_bh_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56ฯ
AssignVariableOp_56AssignVariableOpGassignvariableop_56_adam_seq_self_attention_seq_self_attention_add_wa_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57ฯ
AssignVariableOp_57AssignVariableOpGassignvariableop_57_adam_seq_self_attention_seq_self_attention_add_ba_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58ธ
AssignVariableOp_58AssignVariableOp0assignvariableop_58_adam_lstm_lstm_cell_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59ย
AssignVariableOp_59AssignVariableOp:assignvariableop_59_adam_lstm_lstm_cell_recurrent_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60ถ
AssignVariableOp_60AssignVariableOp.assignvariableop_60_adam_lstm_lstm_cell_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61บ
AssignVariableOp_61AssignVariableOp2assignvariableop_61_adam_time_distributed_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62ธ
AssignVariableOp_62AssignVariableOp0assignvariableop_62_adam_time_distributed_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63ผ
AssignVariableOp_63AssignVariableOp4assignvariableop_63_adam_time_distributed_1_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64บ
AssignVariableOp_64AssignVariableOp2assignvariableop_64_adam_time_distributed_1_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65ผ
AssignVariableOp_65AssignVariableOp4assignvariableop_65_adam_time_distributed_2_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66บ
AssignVariableOp_66AssignVariableOp2assignvariableop_66_adam_time_distributed_2_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_669
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_67
Identity_68IdentityIdentity_67:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_68"#
identity_68Identity_68:output:0*ฃ
_input_shapes
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ๅ

C__inference_lstm_cell_layer_call_and_return_conditional_losses_8203

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimฟ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????
2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????
2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????
2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????
2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????
2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????
2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????
2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????
2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????
2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????
2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:?????????:?????????
:?????????
::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/1
ั

)__inference_sequential_layer_call_fn_7000

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityขStatefulPartitionedCall๔
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_60902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs

๙
sequential_lstm_while_cond_3822<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3>
:sequential_lstm_while_less_sequential_lstm_strided_slice_1R
Nsequential_lstm_while_sequential_lstm_while_cond_3822___redundant_placeholder0R
Nsequential_lstm_while_sequential_lstm_while_cond_3822___redundant_placeholder1R
Nsequential_lstm_while_sequential_lstm_while_cond_3822___redundant_placeholder2R
Nsequential_lstm_while_sequential_lstm_while_cond_3822___redundant_placeholder3"
sequential_lstm_while_identity
ภ
sequential/lstm/while/LessLess!sequential_lstm_while_placeholder:sequential_lstm_while_less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: 2
sequential/lstm/while/Less
sequential/lstm/while/IdentityIdentitysequential/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2 
sequential/lstm/while/Identity"I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?

C__inference_lstm_cell_layer_call_and_return_conditional_losses_4173

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimฟ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????
2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????
2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????
2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????
2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????
2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????
2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????
2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????
2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????
2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????
2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:?????????:?????????
:?????????
::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_namestates:OK
'
_output_shapes
:?????????

 
_user_specified_namestates
ช
ง
?__inference_dense_layer_call_and_return_conditional_losses_4732

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
ช>
๗
lstm_while_body_6302&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_09
5lstm_while_lstm_cell_matmul_readvariableop_resource_0;
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor7
3lstm_while_lstm_cell_matmul_readvariableop_resource9
5lstm_while_lstm_cell_matmul_1_readvariableop_resource8
4lstm_while_lstm_cell_biasadd_readvariableop_resourceอ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape๒
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemฯ
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	(*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpแ
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/while/lstm_cell/MatMulิ
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpส
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/while/lstm_cell/MatMul_1ฟ
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm/while/lstm_cell/addอ
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpฬ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/while/lstm_cell/BiasAddz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/Const
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dim
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm/while/lstm_cell/split
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/Sigmoidข
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2 
lstm/while/lstm_cell/Sigmoid_1ซ
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/mul
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/Reluผ
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/mul_1ฑ
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/add_1ข
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2 
lstm/while/lstm_cell/Sigmoid_2
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/Relu_1ภ
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/mul_2๖
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1m
lstm/while/IdentityIdentitylstm/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm/while/Identity_1o
lstm/while/Identity_2Identitylstm/while/add:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity_2
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:?????????
2
lstm/while/Identity_4
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"ผ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
๒7
ๆ
while_body_7482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceร
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeิ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemภ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	(*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpอ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMulล
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpถ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMul_1ซ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/addพ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpธ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Reluจ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Relu_1ฌ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
์
็ 
__inference__traced_save_8553
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableopK
Gsavev2_seq_self_attention_seq_self_attention_add_wt_read_readvariableopK
Gsavev2_seq_self_attention_seq_self_attention_add_wx_read_readvariableopK
Gsavev2_seq_self_attention_seq_self_attention_add_bh_read_readvariableopK
Gsavev2_seq_self_attention_seq_self_attention_add_wa_read_readvariableopK
Gsavev2_seq_self_attention_seq_self_attention_add_ba_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop8
4savev2_time_distributed_1_kernel_read_readvariableop6
2savev2_time_distributed_1_bias_read_readvariableop8
4savev2_time_distributed_2_kernel_read_readvariableop6
2savev2_time_distributed_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_wt_m_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_wx_m_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_bh_m_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_wa_m_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_ba_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop=
9savev2_adam_time_distributed_kernel_m_read_readvariableop;
7savev2_adam_time_distributed_bias_m_read_readvariableop?
;savev2_adam_time_distributed_1_kernel_m_read_readvariableop=
9savev2_adam_time_distributed_1_bias_m_read_readvariableop?
;savev2_adam_time_distributed_2_kernel_m_read_readvariableop=
9savev2_adam_time_distributed_2_bias_m_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_wt_v_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_wx_v_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_bh_v_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_wa_v_read_readvariableopR
Nsavev2_adam_seq_self_attention_seq_self_attention_add_ba_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop=
9savev2_adam_time_distributed_kernel_v_read_readvariableop;
7savev2_adam_time_distributed_bias_v_read_readvariableop?
;savev2_adam_time_distributed_1_kernel_v_read_readvariableop=
9savev2_adam_time_distributed_1_bias_v_read_readvariableop?
;savev2_adam_time_distributed_2_kernel_v_read_readvariableop=
9savev2_adam_time_distributed_2_bias_v_read_readvariableop
savev2_const

identity_1ขMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_57b570a894614579b6c06d121745c337/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardฆ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameํ'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?&
value๕&B๒&DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_Wt/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_Wx/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_bh/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_Wa/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-4/seq_self_attention_Add_ba/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wt/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wx/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wa/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_ba/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wt/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wx/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_Wa/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBelayer_with_weights-4/seq_self_attention_Add_ba/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices๎
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableopGsavev2_seq_self_attention_seq_self_attention_add_wt_read_readvariableopGsavev2_seq_self_attention_seq_self_attention_add_wx_read_readvariableopGsavev2_seq_self_attention_seq_self_attention_add_bh_read_readvariableopGsavev2_seq_self_attention_seq_self_attention_add_wa_read_readvariableopGsavev2_seq_self_attention_seq_self_attention_add_ba_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop4savev2_time_distributed_1_kernel_read_readvariableop2savev2_time_distributed_1_bias_read_readvariableop4savev2_time_distributed_2_kernel_read_readvariableop2savev2_time_distributed_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_wt_m_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_wx_m_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_bh_m_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_wa_m_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_ba_m_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop9savev2_adam_time_distributed_kernel_m_read_readvariableop7savev2_adam_time_distributed_bias_m_read_readvariableop;savev2_adam_time_distributed_1_kernel_m_read_readvariableop9savev2_adam_time_distributed_1_bias_m_read_readvariableop;savev2_adam_time_distributed_2_kernel_m_read_readvariableop9savev2_adam_time_distributed_2_bias_m_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_wt_v_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_wx_v_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_bh_v_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_wa_v_read_readvariableopNsavev2_adam_seq_self_attention_seq_self_attention_add_ba_v_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop9savev2_adam_time_distributed_kernel_v_read_readvariableop7savev2_adam_time_distributed_bias_v_read_readvariableop;savev2_adam_time_distributed_1_kernel_v_read_readvariableop9savev2_adam_time_distributed_1_bias_v_read_readvariableop;savev2_adam_time_distributed_2_kernel_v_read_readvariableop9savev2_adam_time_distributed_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	2
SaveV2บ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesก
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ไ
_input_shapesา
ฯ: :::::::
 :
 : : :: : : : : :	(:
(:(:	
::
::	:: : :::::::
 :
 : : ::	(:
(:(:	
::
::	::::::::
 :
 : : ::	(:
(:(:	
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::$ 

_output_shapes

:
 :$ 

_output_shapes

:
 : 	

_output_shapes
: :$
 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	(:$ 

_output_shapes

:
(: 

_output_shapes
:(:%!

_output_shapes
:	
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :)%
#
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::* &
$
_output_shapes
::!!

_output_shapes	
::$" 

_output_shapes

:
 :$# 

_output_shapes

:
 : $

_output_shapes
: :$% 

_output_shapes

: : &

_output_shapes
::%'!

_output_shapes
:	(:$( 

_output_shapes

:
(: )

_output_shapes
:(:%*!

_output_shapes
:	
:!+

_output_shapes	
::&,"
 
_output_shapes
:
:!-

_output_shapes	
::%.!

_output_shapes
:	: /

_output_shapes
::)0%
#
_output_shapes
::!1

_output_shapes	
::*2&
$
_output_shapes
::!3

_output_shapes	
::*4&
$
_output_shapes
::!5

_output_shapes	
::$6 

_output_shapes

:
 :$7 

_output_shapes

:
 : 8

_output_shapes
: :$9 

_output_shapes

: : :

_output_shapes
::%;!

_output_shapes
:	(:$< 

_output_shapes

:
(: =

_output_shapes
:(:%>!

_output_shapes
:	
:!?

_output_shapes	
::&@"
 
_output_shapes
:
:!A

_output_shapes	
::%B!

_output_shapes
:	: C

_output_shapes
::D

_output_shapes
: 
ฅ

1__inference_time_distributed_1_layer_call_fn_8110

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_49472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ฺ
{
&__inference_dense_1_layer_call_fn_8310

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_48542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
์
|
'__inference_conv1d_1_layer_call_fn_7050

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๗
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_51272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ญV
๔
>__inference_lstm_layer_call_and_return_conditional_losses_7392
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityขwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shapeฒ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2ฟ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape๘
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2ฌ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02!
lstm_cell/MatMul/ReadVariableOpฃ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMulฑ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/addช
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim็
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_1
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shapeธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterแ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_7307*
condR
while_cond_7306*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
whileต
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape๑
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permฎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::2
whilewhile:_ [
5
_output_shapes#
!:??????????????????
"
_user_specified_name
inputs/0
ฑZ
?
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_7849

inputs#
shape_3_readvariableop_resource#
shape_5_readvariableop_resource!
add_1_readvariableop_resource#
shape_7_readvariableop_resource!
add_2_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2๎
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2H
Shape_2Shapeinputs*
T0*
_output_shapes
:2	
Shape_2^
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"
       2	
Shape_3`
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????
2	
Reshape
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:
 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:
 2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2ข
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_2b
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsReshape_2:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2

ExpandDimsH
Shape_4Shapeinputs*
T0*
_output_shapes
:2	
Shape_4b
	unstack_2UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02
Shape_5/ReadVariableOpc
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"
       2	
Shape_5`
	unstack_3UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape_3/shapeu
	Reshape_3ReshapeinputsReshape_3/shape:output:0*
T0*'
_output_shapes
:?????????
2
	Reshape_3
transpose_1/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:
 2
transpose_1s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:
 2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_5/shape/2ค
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_5f
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsReshape_5:output:0ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:????????? 2
ExpandDims_1y
addAddV2ExpandDims:output:0ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? 2
add
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
add_1/ReadVariableOpx
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
add_1Y
TanhTanh	add_1:z:0*
T0*/
_output_shapes
:????????? 2
TanhJ
Shape_6ShapeTanh:y:0*
T0*
_output_shapes
:2	
Shape_6d
	unstack_4UnpackShape_6:output:0*
T0*
_output_shapes

: : : : *	
num2
	unstack_4
Shape_7/ReadVariableOpReadVariableOpshape_7_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_7/ReadVariableOpc
Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_7`
	unstack_5UnpackShape_7:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_5s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape_6/shapew
	Reshape_6ReshapeTanh:y:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:????????? 2
	Reshape_6
transpose_2/ReadVariableOpReadVariableOpshape_7_readvariableop_resource*
_output_shapes

: *
dtype02
transpose_2/ReadVariableOpu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_7/shapeu
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_7x
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2h
Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/1h
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/2h
Reshape_8/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/3ภ
Reshape_8/shapePackunstack_4:output:0Reshape_8/shape/1:output:0Reshape_8/shape/2:output:0Reshape_8/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_8/shape
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*/
_output_shapes
:?????????2
	Reshape_8
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
add_2/ReadVariableOp
add_2AddV2Reshape_8:output:0add_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
add_2ฆ
Reshape_9/shapePackstrided_slice_1:output:0strided_slice_2:output:0strided_slice_2:output:0*
N*
T0*
_output_shapes
:2
Reshape_9/shape
	Reshape_9Reshape	add_2:z:0Reshape_9/shape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Reshape_9p
ReluReluReshape_9:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
Reluy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices
MaxMaxRelu:activations:0Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Max{
subSubRelu:activations:0Max:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
subb
ExpExpsub:z:0*
T0*=
_output_shapes+
):'???????????????????????????2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Sum|
truedivRealDivExp:y:0Sum:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
truedivy
MatMul_3BatchMatMulV2truediv:z:0inputs*
T0*4
_output_shapes"
 :??????????????????
2

MatMul_3S
Shape_8ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_8r
IdentityIdentityMatMul_3:output:0*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
๎U
๒
>__inference_lstm_layer_call_and_return_conditional_losses_7720

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityขwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shapeฒ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2ฟ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape๘
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2ฌ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02!
lstm_cell/MatMul/ReadVariableOpฃ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMulฑ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/addช
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim็
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_1
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shapeธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterแ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_7635*
condR
while_cond_7634*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
whileต
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape่
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permฅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::2
whilewhile:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
์ณ

__inference__wrapped_model_4070
conv1d_inputA
=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource5
1sequential_conv1d_biasadd_readvariableop_resourceC
?sequential_conv1d_1_conv1d_expanddims_1_readvariableop_resource7
3sequential_conv1d_1_biasadd_readvariableop_resourceC
?sequential_conv1d_2_conv1d_expanddims_1_readvariableop_resource7
3sequential_conv1d_2_biasadd_readvariableop_resource<
8sequential_lstm_lstm_cell_matmul_readvariableop_resource>
:sequential_lstm_lstm_cell_matmul_1_readvariableop_resource=
9sequential_lstm_lstm_cell_biasadd_readvariableop_resourceA
=sequential_seq_self_attention_shape_3_readvariableop_resourceA
=sequential_seq_self_attention_shape_5_readvariableop_resource?
;sequential_seq_self_attention_add_1_readvariableop_resourceA
=sequential_seq_self_attention_shape_7_readvariableop_resource?
;sequential_seq_self_attention_add_2_readvariableop_resourceD
@sequential_time_distributed_dense_matmul_readvariableop_resourceE
Asequential_time_distributed_dense_biasadd_readvariableop_resourceH
Dsequential_time_distributed_1_dense_1_matmul_readvariableop_resourceI
Esequential_time_distributed_1_dense_1_biasadd_readvariableop_resourceH
Dsequential_time_distributed_2_dense_2_matmul_readvariableop_resourceI
Esequential_time_distributed_2_dense_2_biasadd_readvariableop_resource
identityขsequential/lstm/while
'sequential/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'sequential/conv1d/conv1d/ExpandDims/dimา
#sequential/conv1d/conv1d/ExpandDims
ExpandDimsconv1d_input0sequential/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2%
#sequential/conv1d/conv1d/ExpandDims๏
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype026
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp
)sequential/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/conv1d/conv1d/ExpandDims_1/dim
%sequential/conv1d/conv1d/ExpandDims_1
ExpandDims<sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:02sequential/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:2'
%sequential/conv1d/conv1d/ExpandDims_1
sequential/conv1d/conv1dConv2D,sequential/conv1d/conv1d/ExpandDims:output:0.sequential/conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
sequential/conv1d/conv1dษ
 sequential/conv1d/conv1d/SqueezeSqueeze!sequential/conv1d/conv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2"
 sequential/conv1d/conv1d/Squeezeร
(sequential/conv1d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(sequential/conv1d/BiasAdd/ReadVariableOpี
sequential/conv1d/BiasAddBiasAdd)sequential/conv1d/conv1d/Squeeze:output:00sequential/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2
sequential/conv1d/BiasAdd
sequential/conv1d/ReluRelu"sequential/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????2
sequential/conv1d/Reluก
)sequential/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)sequential/conv1d_1/conv1d/ExpandDims/dim๑
%sequential/conv1d_1/conv1d/ExpandDims
ExpandDims$sequential/conv1d/Relu:activations:02sequential/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2'
%sequential/conv1d_1/conv1d/ExpandDims๖
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?sequential_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype028
6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
+sequential/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/conv1d_1/conv1d/ExpandDims_1/dim
'sequential/conv1d_1/conv1d/ExpandDims_1
ExpandDims>sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:04sequential/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2)
'sequential/conv1d_1/conv1d/ExpandDims_1
sequential/conv1d_1/conv1dConv2D.sequential/conv1d_1/conv1d/ExpandDims:output:00sequential/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
sequential/conv1d_1/conv1dฯ
"sequential/conv1d_1/conv1d/SqueezeSqueeze#sequential/conv1d_1/conv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2$
"sequential/conv1d_1/conv1d/Squeezeษ
*sequential/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*sequential/conv1d_1/BiasAdd/ReadVariableOp?
sequential/conv1d_1/BiasAddBiasAdd+sequential/conv1d_1/conv1d/Squeeze:output:02sequential/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2
sequential/conv1d_1/BiasAdd
sequential/conv1d_1/ReluRelu$sequential/conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:?????????2
sequential/conv1d_1/Reluก
)sequential/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)sequential/conv1d_2/conv1d/ExpandDims/dim๓
%sequential/conv1d_2/conv1d/ExpandDims
ExpandDims&sequential/conv1d_1/Relu:activations:02sequential/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2'
%sequential/conv1d_2/conv1d/ExpandDims๖
6sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?sequential_conv1d_2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype028
6sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
+sequential/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/conv1d_2/conv1d/ExpandDims_1/dim
'sequential/conv1d_2/conv1d/ExpandDims_1
ExpandDims>sequential/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:04sequential/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2)
'sequential/conv1d_2/conv1d/ExpandDims_1
sequential/conv1d_2/conv1dConv2D.sequential/conv1d_2/conv1d/ExpandDims:output:00sequential/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
sequential/conv1d_2/conv1dฯ
"sequential/conv1d_2/conv1d/SqueezeSqueeze#sequential/conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2$
"sequential/conv1d_2/conv1d/Squeezeษ
*sequential/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*sequential/conv1d_2/BiasAdd/ReadVariableOp?
sequential/conv1d_2/BiasAddBiasAdd+sequential/conv1d_2/conv1d/Squeeze:output:02sequential/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2
sequential/conv1d_2/BiasAdd
sequential/conv1d_2/ReluRelu$sequential/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????2
sequential/conv1d_2/Relu
'sequential/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential/max_pooling1d/ExpandDims/dimํ
#sequential/max_pooling1d/ExpandDims
ExpandDims&sequential/conv1d_2/Relu:activations:00sequential/max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2%
#sequential/max_pooling1d/ExpandDims๋
 sequential/max_pooling1d/MaxPoolMaxPool,sequential/max_pooling1d/ExpandDims:output:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling1d/MaxPoolศ
 sequential/max_pooling1d/SqueezeSqueeze)sequential/max_pooling1d/MaxPool:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims
2"
 sequential/max_pooling1d/Squeeze
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/flatten/Constฤ
sequential/flatten/ReshapeReshape)sequential/max_pooling1d/Squeeze:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
sequential/flatten/Reshape
'sequential/repeat_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential/repeat_vector/ExpandDims/dimๆ
#sequential/repeat_vector/ExpandDims
ExpandDims#sequential/flatten/Reshape:output:00sequential/repeat_vector/ExpandDims/dim:output:0*
T0*,
_output_shapes
:?????????2%
#sequential/repeat_vector/ExpandDims
sequential/repeat_vector/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2 
sequential/repeat_vector/stackิ
sequential/repeat_vector/TileTile,sequential/repeat_vector/ExpandDims:output:0'sequential/repeat_vector/stack:output:0*
T0*,
_output_shapes
:?????????2
sequential/repeat_vector/Tile
sequential/lstm/ShapeShape&sequential/repeat_vector/Tile:output:0*
T0*
_output_shapes
:2
sequential/lstm/Shape
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/lstm/strided_slice/stack
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_1
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_2ย
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/lstm/strided_slice|
sequential/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
sequential/lstm/zeros/mul/yฌ
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/mul
sequential/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
sequential/lstm/zeros/Less/yง
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/Less
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2 
sequential/lstm/zeros/packed/1ร
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/lstm/zeros/packed
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros/Constต
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/zeros
sequential/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
sequential/lstm/zeros_1/mul/yฒ
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/mul
sequential/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2 
sequential/lstm/zeros_1/Less/yฏ
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/Less
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2"
 sequential/lstm/zeros_1/packed/1ษ
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
sequential/lstm/zeros_1/packed
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros_1/Constฝ
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/zeros_1
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
sequential/lstm/transpose/permห
sequential/lstm/transpose	Transpose&sequential/repeat_vector/Tile:output:0'sequential/lstm/transpose/perm:output:0*
T0*,
_output_shapes
:?????????2
sequential/lstm/transpose
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:2
sequential/lstm/Shape_1
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_1/stack
'sequential/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_1
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_2ฮ
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
sequential/lstm/strided_slice_1ฅ
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential/lstm/TensorArrayV2/element_shape๒
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
sequential/lstm/TensorArrayV2฿
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeธ
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7sequential/lstm/TensorArrayUnstack/TensorListFromTensor
%sequential/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_2/stack
'sequential/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_1
'sequential/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_2?
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2!
sequential/lstm/strided_slice_2?
/sequential/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp8sequential_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype021
/sequential/lstm/lstm_cell/MatMul/ReadVariableOpใ
 sequential/lstm/lstm_cell/MatMulMatMul(sequential/lstm/strided_slice_2:output:07sequential/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2"
 sequential/lstm/lstm_cell/MatMulแ
1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:sequential_lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype023
1sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp฿
"sequential/lstm/lstm_cell/MatMul_1MatMulsequential/lstm/zeros:output:09sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2$
"sequential/lstm/lstm_cell/MatMul_1ำ
sequential/lstm/lstm_cell/addAddV2*sequential/lstm/lstm_cell/MatMul:product:0,sequential/lstm/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
sequential/lstm/lstm_cell/addฺ
0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9sequential_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype022
0sequential/lstm/lstm_cell/BiasAdd/ReadVariableOpเ
!sequential/lstm/lstm_cell/BiasAddBiasAdd!sequential/lstm/lstm_cell/add:z:08sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2#
!sequential/lstm/lstm_cell/BiasAdd
sequential/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential/lstm/lstm_cell/Const
)sequential/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential/lstm/lstm_cell/split/split_dimง
sequential/lstm/lstm_cell/splitSplit2sequential/lstm/lstm_cell/split/split_dim:output:0*sequential/lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2!
sequential/lstm/lstm_cell/splitญ
!sequential/lstm/lstm_cell/SigmoidSigmoid(sequential/lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2#
!sequential/lstm/lstm_cell/Sigmoidฑ
#sequential/lstm/lstm_cell/Sigmoid_1Sigmoid(sequential/lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2%
#sequential/lstm/lstm_cell/Sigmoid_1ย
sequential/lstm/lstm_cell/mulMul'sequential/lstm/lstm_cell/Sigmoid_1:y:0 sequential/lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/lstm_cell/mulค
sequential/lstm/lstm_cell/ReluRelu(sequential/lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2 
sequential/lstm/lstm_cell/Reluะ
sequential/lstm/lstm_cell/mul_1Mul%sequential/lstm/lstm_cell/Sigmoid:y:0,sequential/lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2!
sequential/lstm/lstm_cell/mul_1ล
sequential/lstm/lstm_cell/add_1AddV2!sequential/lstm/lstm_cell/mul:z:0#sequential/lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2!
sequential/lstm/lstm_cell/add_1ฑ
#sequential/lstm/lstm_cell/Sigmoid_2Sigmoid(sequential/lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2%
#sequential/lstm/lstm_cell/Sigmoid_2ฃ
 sequential/lstm/lstm_cell/Relu_1Relu#sequential/lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2"
 sequential/lstm/lstm_cell/Relu_1ิ
sequential/lstm/lstm_cell/mul_2Mul'sequential/lstm/lstm_cell/Sigmoid_2:y:0.sequential/lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2!
sequential/lstm/lstm_cell/mul_2ฏ
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2/
-sequential/lstm/TensorArrayV2_1/element_shape๘
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
sequential/lstm/TensorArrayV2_1n
sequential/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/lstm/time
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(sequential/lstm/while/maximum_iterations
"sequential/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential/lstm/while/loop_counterั
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_lstm_lstm_cell_matmul_readvariableop_resource:sequential_lstm_lstm_cell_matmul_1_readvariableop_resource9sequential_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
sequential_lstm_while_body_3823*+
cond#R!
sequential_lstm_while_cond_3822*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
sequential/lstm/whileี
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2B
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeจ
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype024
2sequential/lstm/TensorArrayV2Stack/TensorListStackก
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2'
%sequential/lstm/strided_slice_3/stack
'sequential/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential/lstm/strided_slice_3/stack_1
'sequential/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_3/stack_2๚
sequential/lstm/strided_slice_3StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2!
sequential/lstm/strided_slice_3
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 sequential/lstm/transpose_1/permๅ
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2
sequential/lstm/transpose_1
sequential/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/runtime
#sequential/seq_self_attention/ShapeShapesequential/lstm/transpose_1:y:0*
T0*
_output_shapes
:2%
#sequential/seq_self_attention/Shapeฐ
1sequential/seq_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1sequential/seq_self_attention/strided_slice/stackด
3sequential/seq_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/seq_self_attention/strided_slice/stack_1ด
3sequential/seq_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/seq_self_attention/strided_slice/stack_2
+sequential/seq_self_attention/strided_sliceStridedSlice,sequential/seq_self_attention/Shape:output:0:sequential/seq_self_attention/strided_slice/stack:output:0<sequential/seq_self_attention/strided_slice/stack_1:output:0<sequential/seq_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/seq_self_attention/strided_slice
%sequential/seq_self_attention/Shape_1Shapesequential/lstm/transpose_1:y:0*
T0*
_output_shapes
:2'
%sequential/seq_self_attention/Shape_1ด
3sequential/seq_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/seq_self_attention/strided_slice_1/stackธ
5sequential/seq_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/seq_self_attention/strided_slice_1/stack_1ธ
5sequential/seq_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/seq_self_attention/strided_slice_1/stack_2ข
-sequential/seq_self_attention/strided_slice_1StridedSlice.sequential/seq_self_attention/Shape_1:output:0<sequential/seq_self_attention/strided_slice_1/stack:output:0>sequential/seq_self_attention/strided_slice_1/stack_1:output:0>sequential/seq_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/seq_self_attention/strided_slice_1ด
3sequential/seq_self_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/seq_self_attention/strided_slice_2/stackธ
5sequential/seq_self_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/seq_self_attention/strided_slice_2/stack_1ธ
5sequential/seq_self_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/seq_self_attention/strided_slice_2/stack_2ข
-sequential/seq_self_attention/strided_slice_2StridedSlice.sequential/seq_self_attention/Shape_1:output:0<sequential/seq_self_attention/strided_slice_2/stack:output:0>sequential/seq_self_attention/strided_slice_2/stack_1:output:0>sequential/seq_self_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/seq_self_attention/strided_slice_2
%sequential/seq_self_attention/Shape_2Shapesequential/lstm/transpose_1:y:0*
T0*
_output_shapes
:2'
%sequential/seq_self_attention/Shape_2ธ
%sequential/seq_self_attention/unstackUnpack.sequential/seq_self_attention/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2'
%sequential/seq_self_attention/unstack๊
4sequential/seq_self_attention/Shape_3/ReadVariableOpReadVariableOp=sequential_seq_self_attention_shape_3_readvariableop_resource*
_output_shapes

:
 *
dtype026
4sequential/seq_self_attention/Shape_3/ReadVariableOp
%sequential/seq_self_attention/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"
       2'
%sequential/seq_self_attention/Shape_3บ
'sequential/seq_self_attention/unstack_1Unpack.sequential/seq_self_attention/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2)
'sequential/seq_self_attention/unstack_1ซ
+sequential/seq_self_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2-
+sequential/seq_self_attention/Reshape/shapeโ
%sequential/seq_self_attention/ReshapeReshapesequential/lstm/transpose_1:y:04sequential/seq_self_attention/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2'
%sequential/seq_self_attention/Reshape๎
6sequential/seq_self_attention/transpose/ReadVariableOpReadVariableOp=sequential_seq_self_attention_shape_3_readvariableop_resource*
_output_shapes

:
 *
dtype028
6sequential/seq_self_attention/transpose/ReadVariableOpญ
,sequential/seq_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2.
,sequential/seq_self_attention/transpose/perm?
'sequential/seq_self_attention/transpose	Transpose>sequential/seq_self_attention/transpose/ReadVariableOp:value:05sequential/seq_self_attention/transpose/perm:output:0*
T0*
_output_shapes

:
 2)
'sequential/seq_self_attention/transposeฏ
-sequential/seq_self_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2/
-sequential/seq_self_attention/Reshape_1/shape๋
'sequential/seq_self_attention/Reshape_1Reshape+sequential/seq_self_attention/transpose:y:06sequential/seq_self_attention/Reshape_1/shape:output:0*
T0*
_output_shapes

:
 2)
'sequential/seq_self_attention/Reshape_1๊
$sequential/seq_self_attention/MatMulMatMul.sequential/seq_self_attention/Reshape:output:00sequential/seq_self_attention/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2&
$sequential/seq_self_attention/MatMulค
/sequential/seq_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential/seq_self_attention/Reshape_2/shape/1ค
/sequential/seq_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 21
/sequential/seq_self_attention/Reshape_2/shape/2ธ
-sequential/seq_self_attention/Reshape_2/shapePack.sequential/seq_self_attention/unstack:output:08sequential/seq_self_attention/Reshape_2/shape/1:output:08sequential/seq_self_attention/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2/
-sequential/seq_self_attention/Reshape_2/shape๛
'sequential/seq_self_attention/Reshape_2Reshape.sequential/seq_self_attention/MatMul:product:06sequential/seq_self_attention/Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2)
'sequential/seq_self_attention/Reshape_2
,sequential/seq_self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential/seq_self_attention/ExpandDims/dim
(sequential/seq_self_attention/ExpandDims
ExpandDims0sequential/seq_self_attention/Reshape_2:output:05sequential/seq_self_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2*
(sequential/seq_self_attention/ExpandDims
%sequential/seq_self_attention/Shape_4Shapesequential/lstm/transpose_1:y:0*
T0*
_output_shapes
:2'
%sequential/seq_self_attention/Shape_4ผ
'sequential/seq_self_attention/unstack_2Unpack.sequential/seq_self_attention/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num2)
'sequential/seq_self_attention/unstack_2๊
4sequential/seq_self_attention/Shape_5/ReadVariableOpReadVariableOp=sequential_seq_self_attention_shape_5_readvariableop_resource*
_output_shapes

:
 *
dtype026
4sequential/seq_self_attention/Shape_5/ReadVariableOp
%sequential/seq_self_attention/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"
       2'
%sequential/seq_self_attention/Shape_5บ
'sequential/seq_self_attention/unstack_3Unpack.sequential/seq_self_attention/Shape_5:output:0*
T0*
_output_shapes
: : *	
num2)
'sequential/seq_self_attention/unstack_3ฏ
-sequential/seq_self_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2/
-sequential/seq_self_attention/Reshape_3/shape่
'sequential/seq_self_attention/Reshape_3Reshapesequential/lstm/transpose_1:y:06sequential/seq_self_attention/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????
2)
'sequential/seq_self_attention/Reshape_3๒
8sequential/seq_self_attention/transpose_1/ReadVariableOpReadVariableOp=sequential_seq_self_attention_shape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02:
8sequential/seq_self_attention/transpose_1/ReadVariableOpฑ
.sequential/seq_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential/seq_self_attention/transpose_1/perm
)sequential/seq_self_attention/transpose_1	Transpose@sequential/seq_self_attention/transpose_1/ReadVariableOp:value:07sequential/seq_self_attention/transpose_1/perm:output:0*
T0*
_output_shapes

:
 2+
)sequential/seq_self_attention/transpose_1ฏ
-sequential/seq_self_attention/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2/
-sequential/seq_self_attention/Reshape_4/shapeํ
'sequential/seq_self_attention/Reshape_4Reshape-sequential/seq_self_attention/transpose_1:y:06sequential/seq_self_attention/Reshape_4/shape:output:0*
T0*
_output_shapes

:
 2)
'sequential/seq_self_attention/Reshape_4๐
&sequential/seq_self_attention/MatMul_1MatMul0sequential/seq_self_attention/Reshape_3:output:00sequential/seq_self_attention/Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2(
&sequential/seq_self_attention/MatMul_1ค
/sequential/seq_self_attention/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential/seq_self_attention/Reshape_5/shape/1ค
/sequential/seq_self_attention/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 21
/sequential/seq_self_attention/Reshape_5/shape/2บ
-sequential/seq_self_attention/Reshape_5/shapePack0sequential/seq_self_attention/unstack_2:output:08sequential/seq_self_attention/Reshape_5/shape/1:output:08sequential/seq_self_attention/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2/
-sequential/seq_self_attention/Reshape_5/shape?
'sequential/seq_self_attention/Reshape_5Reshape0sequential/seq_self_attention/MatMul_1:product:06sequential/seq_self_attention/Reshape_5/shape:output:0*
T0*+
_output_shapes
:????????? 2)
'sequential/seq_self_attention/Reshape_5ข
.sequential/seq_self_attention/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential/seq_self_attention/ExpandDims_1/dim
*sequential/seq_self_attention/ExpandDims_1
ExpandDims0sequential/seq_self_attention/Reshape_5:output:07sequential/seq_self_attention/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:????????? 2,
*sequential/seq_self_attention/ExpandDims_1๑
!sequential/seq_self_attention/addAddV21sequential/seq_self_attention/ExpandDims:output:03sequential/seq_self_attention/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? 2#
!sequential/seq_self_attention/addเ
2sequential/seq_self_attention/add_1/ReadVariableOpReadVariableOp;sequential_seq_self_attention_add_1_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential/seq_self_attention/add_1/ReadVariableOp๐
#sequential/seq_self_attention/add_1AddV2%sequential/seq_self_attention/add:z:0:sequential/seq_self_attention/add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2%
#sequential/seq_self_attention/add_1ณ
"sequential/seq_self_attention/TanhTanh'sequential/seq_self_attention/add_1:z:0*
T0*/
_output_shapes
:????????? 2$
"sequential/seq_self_attention/Tanhค
%sequential/seq_self_attention/Shape_6Shape&sequential/seq_self_attention/Tanh:y:0*
T0*
_output_shapes
:2'
%sequential/seq_self_attention/Shape_6พ
'sequential/seq_self_attention/unstack_4Unpack.sequential/seq_self_attention/Shape_6:output:0*
T0*
_output_shapes

: : : : *	
num2)
'sequential/seq_self_attention/unstack_4๊
4sequential/seq_self_attention/Shape_7/ReadVariableOpReadVariableOp=sequential_seq_self_attention_shape_7_readvariableop_resource*
_output_shapes

: *
dtype026
4sequential/seq_self_attention/Shape_7/ReadVariableOp
%sequential/seq_self_attention/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential/seq_self_attention/Shape_7บ
'sequential/seq_self_attention/unstack_5Unpack.sequential/seq_self_attention/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2)
'sequential/seq_self_attention/unstack_5ฏ
-sequential/seq_self_attention/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2/
-sequential/seq_self_attention/Reshape_6/shape๏
'sequential/seq_self_attention/Reshape_6Reshape&sequential/seq_self_attention/Tanh:y:06sequential/seq_self_attention/Reshape_6/shape:output:0*
T0*'
_output_shapes
:????????? 2)
'sequential/seq_self_attention/Reshape_6๒
8sequential/seq_self_attention/transpose_2/ReadVariableOpReadVariableOp=sequential_seq_self_attention_shape_7_readvariableop_resource*
_output_shapes

: *
dtype02:
8sequential/seq_self_attention/transpose_2/ReadVariableOpฑ
.sequential/seq_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential/seq_self_attention/transpose_2/perm
)sequential/seq_self_attention/transpose_2	Transpose@sequential/seq_self_attention/transpose_2/ReadVariableOp:value:07sequential/seq_self_attention/transpose_2/perm:output:0*
T0*
_output_shapes

: 2+
)sequential/seq_self_attention/transpose_2ฏ
-sequential/seq_self_attention/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2/
-sequential/seq_self_attention/Reshape_7/shapeํ
'sequential/seq_self_attention/Reshape_7Reshape-sequential/seq_self_attention/transpose_2:y:06sequential/seq_self_attention/Reshape_7/shape:output:0*
T0*
_output_shapes

: 2)
'sequential/seq_self_attention/Reshape_7๐
&sequential/seq_self_attention/MatMul_2MatMul0sequential/seq_self_attention/Reshape_6:output:00sequential/seq_self_attention/Reshape_7:output:0*
T0*'
_output_shapes
:?????????2(
&sequential/seq_self_attention/MatMul_2ค
/sequential/seq_self_attention/Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential/seq_self_attention/Reshape_8/shape/1ค
/sequential/seq_self_attention/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential/seq_self_attention/Reshape_8/shape/2ค
/sequential/seq_self_attention/Reshape_8/shape/3Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential/seq_self_attention/Reshape_8/shape/3๔
-sequential/seq_self_attention/Reshape_8/shapePack0sequential/seq_self_attention/unstack_4:output:08sequential/seq_self_attention/Reshape_8/shape/1:output:08sequential/seq_self_attention/Reshape_8/shape/2:output:08sequential/seq_self_attention/Reshape_8/shape/3:output:0*
N*
T0*
_output_shapes
:2/
-sequential/seq_self_attention/Reshape_8/shape
'sequential/seq_self_attention/Reshape_8Reshape0sequential/seq_self_attention/MatMul_2:product:06sequential/seq_self_attention/Reshape_8/shape:output:0*
T0*/
_output_shapes
:?????????2)
'sequential/seq_self_attention/Reshape_8เ
2sequential/seq_self_attention/add_2/ReadVariableOpReadVariableOp;sequential_seq_self_attention_add_2_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential/seq_self_attention/add_2/ReadVariableOp๛
#sequential/seq_self_attention/add_2AddV20sequential/seq_self_attention/Reshape_8:output:0:sequential/seq_self_attention/add_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2%
#sequential/seq_self_attention/add_2ผ
-sequential/seq_self_attention/Reshape_9/shapePack6sequential/seq_self_attention/strided_slice_1:output:06sequential/seq_self_attention/strided_slice_2:output:06sequential/seq_self_attention/strided_slice_2:output:0*
N*
T0*
_output_shapes
:2/
-sequential/seq_self_attention/Reshape_9/shape
'sequential/seq_self_attention/Reshape_9Reshape'sequential/seq_self_attention/add_2:z:06sequential/seq_self_attention/Reshape_9/shape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2)
'sequential/seq_self_attention/Reshape_9ส
"sequential/seq_self_attention/ReluRelu0sequential/seq_self_attention/Reshape_9:output:0*
T0*=
_output_shapes+
):'???????????????????????????2$
"sequential/seq_self_attention/Reluต
3sequential/seq_self_attention/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential/seq_self_attention/Max/reduction_indices
!sequential/seq_self_attention/MaxMax0sequential/seq_self_attention/Relu:activations:0<sequential/seq_self_attention/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2#
!sequential/seq_self_attention/Max๓
!sequential/seq_self_attention/subSub0sequential/seq_self_attention/Relu:activations:0*sequential/seq_self_attention/Max:output:0*
T0*=
_output_shapes+
):'???????????????????????????2#
!sequential/seq_self_attention/subผ
!sequential/seq_self_attention/ExpExp%sequential/seq_self_attention/sub:z:0*
T0*=
_output_shapes+
):'???????????????????????????2#
!sequential/seq_self_attention/Expต
3sequential/seq_self_attention/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3sequential/seq_self_attention/Sum/reduction_indices
!sequential/seq_self_attention/SumSum%sequential/seq_self_attention/Exp:y:0<sequential/seq_self_attention/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2#
!sequential/seq_self_attention/Sum๔
%sequential/seq_self_attention/truedivRealDiv%sequential/seq_self_attention/Exp:y:0*sequential/seq_self_attention/Sum:output:0*
T0*=
_output_shapes+
):'???????????????????????????2'
%sequential/seq_self_attention/truediv์
&sequential/seq_self_attention/MatMul_3BatchMatMulV2)sequential/seq_self_attention/truediv:z:0sequential/lstm/transpose_1:y:0*
T0*4
_output_shapes"
 :??????????????????
2(
&sequential/seq_self_attention/MatMul_3ญ
%sequential/seq_self_attention/Shape_8Shape/sequential/seq_self_attention/MatMul_3:output:0*
T0*
_output_shapes
:2'
%sequential/seq_self_attention/Shape_8ฅ
!sequential/time_distributed/ShapeShape/sequential/seq_self_attention/MatMul_3:output:0*
T0*
_output_shapes
:2#
!sequential/time_distributed/Shapeฌ
/sequential/time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/sequential/time_distributed/strided_slice/stackฐ
1sequential/time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/time_distributed/strided_slice/stack_1ฐ
1sequential/time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/time_distributed/strided_slice/stack_2
)sequential/time_distributed/strided_sliceStridedSlice*sequential/time_distributed/Shape:output:08sequential/time_distributed/strided_slice/stack:output:0:sequential/time_distributed/strided_slice/stack_1:output:0:sequential/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential/time_distributed/strided_sliceง
)sequential/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2+
)sequential/time_distributed/Reshape/shape์
#sequential/time_distributed/ReshapeReshape/sequential/seq_self_attention/MatMul_3:output:02sequential/time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2%
#sequential/time_distributed/Reshape๔
7sequential/time_distributed/dense/MatMul/ReadVariableOpReadVariableOp@sequential_time_distributed_dense_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype029
7sequential/time_distributed/dense/MatMul/ReadVariableOp
(sequential/time_distributed/dense/MatMulMatMul,sequential/time_distributed/Reshape:output:0?sequential/time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2*
(sequential/time_distributed/dense/MatMul๓
8sequential/time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOpAsequential_time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02:
8sequential/time_distributed/dense/BiasAdd/ReadVariableOp
)sequential/time_distributed/dense/BiasAddBiasAdd2sequential/time_distributed/dense/MatMul:product:0@sequential/time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2+
)sequential/time_distributed/dense/BiasAddฟ
&sequential/time_distributed/dense/ReluRelu2sequential/time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2(
&sequential/time_distributed/dense/Reluฉ
-sequential/time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential/time_distributed/Reshape_1/shape/0ก
-sequential/time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2/
-sequential/time_distributed/Reshape_1/shape/2ด
+sequential/time_distributed/Reshape_1/shapePack6sequential/time_distributed/Reshape_1/shape/0:output:02sequential/time_distributed/strided_slice:output:06sequential/time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+sequential/time_distributed/Reshape_1/shape
%sequential/time_distributed/Reshape_1Reshape4sequential/time_distributed/dense/Relu:activations:04sequential/time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2'
%sequential/time_distributed/Reshape_1ซ
+sequential/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2-
+sequential/time_distributed/Reshape_2/shape๒
%sequential/time_distributed/Reshape_2Reshape/sequential/seq_self_attention/MatMul_3:output:04sequential/time_distributed/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????
2'
%sequential/time_distributed/Reshape_2จ
#sequential/time_distributed_1/ShapeShape.sequential/time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:2%
#sequential/time_distributed_1/Shapeฐ
1sequential/time_distributed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1sequential/time_distributed_1/strided_slice/stackด
3sequential/time_distributed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/time_distributed_1/strided_slice/stack_1ด
3sequential/time_distributed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/time_distributed_1/strided_slice/stack_2
+sequential/time_distributed_1/strided_sliceStridedSlice,sequential/time_distributed_1/Shape:output:0:sequential/time_distributed_1/strided_slice/stack:output:0<sequential/time_distributed_1/strided_slice/stack_1:output:0<sequential/time_distributed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/time_distributed_1/strided_sliceซ
+sequential/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2-
+sequential/time_distributed_1/Reshape/shape๒
%sequential/time_distributed_1/ReshapeReshape.sequential/time_distributed/Reshape_1:output:04sequential/time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2'
%sequential/time_distributed_1/Reshape
;sequential/time_distributed_1/dense_1/MatMul/ReadVariableOpReadVariableOpDsequential_time_distributed_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;sequential/time_distributed_1/dense_1/MatMul/ReadVariableOp
,sequential/time_distributed_1/dense_1/MatMulMatMul.sequential/time_distributed_1/Reshape:output:0Csequential/time_distributed_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2.
,sequential/time_distributed_1/dense_1/MatMul?
<sequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpEsequential_time_distributed_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<sequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOp
-sequential/time_distributed_1/dense_1/BiasAddBiasAdd6sequential/time_distributed_1/dense_1/MatMul:product:0Dsequential/time_distributed_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2/
-sequential/time_distributed_1/dense_1/BiasAddห
*sequential/time_distributed_1/dense_1/ReluRelu6sequential/time_distributed_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2,
*sequential/time_distributed_1/dense_1/Reluญ
/sequential/time_distributed_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential/time_distributed_1/Reshape_1/shape/0ฅ
/sequential/time_distributed_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :21
/sequential/time_distributed_1/Reshape_1/shape/2พ
-sequential/time_distributed_1/Reshape_1/shapePack8sequential/time_distributed_1/Reshape_1/shape/0:output:04sequential/time_distributed_1/strided_slice:output:08sequential/time_distributed_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2/
-sequential/time_distributed_1/Reshape_1/shape
'sequential/time_distributed_1/Reshape_1Reshape8sequential/time_distributed_1/dense_1/Relu:activations:06sequential/time_distributed_1/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2)
'sequential/time_distributed_1/Reshape_1ฏ
-sequential/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2/
-sequential/time_distributed_1/Reshape_2/shape๘
'sequential/time_distributed_1/Reshape_2Reshape.sequential/time_distributed/Reshape_1:output:06sequential/time_distributed_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:?????????2)
'sequential/time_distributed_1/Reshape_2ช
#sequential/time_distributed_2/ShapeShape0sequential/time_distributed_1/Reshape_1:output:0*
T0*
_output_shapes
:2%
#sequential/time_distributed_2/Shapeฐ
1sequential/time_distributed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1sequential/time_distributed_2/strided_slice/stackด
3sequential/time_distributed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/time_distributed_2/strided_slice/stack_1ด
3sequential/time_distributed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/time_distributed_2/strided_slice/stack_2
+sequential/time_distributed_2/strided_sliceStridedSlice,sequential/time_distributed_2/Shape:output:0:sequential/time_distributed_2/strided_slice/stack:output:0<sequential/time_distributed_2/strided_slice/stack_1:output:0<sequential/time_distributed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/time_distributed_2/strided_sliceซ
+sequential/time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2-
+sequential/time_distributed_2/Reshape/shape๔
%sequential/time_distributed_2/ReshapeReshape0sequential/time_distributed_1/Reshape_1:output:04sequential/time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2'
%sequential/time_distributed_2/Reshape
;sequential/time_distributed_2/dense_2/MatMul/ReadVariableOpReadVariableOpDsequential_time_distributed_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02=
;sequential/time_distributed_2/dense_2/MatMul/ReadVariableOp
,sequential/time_distributed_2/dense_2/MatMulMatMul.sequential/time_distributed_2/Reshape:output:0Csequential/time_distributed_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,sequential/time_distributed_2/dense_2/MatMul?
<sequential/time_distributed_2/dense_2/BiasAdd/ReadVariableOpReadVariableOpEsequential_time_distributed_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<sequential/time_distributed_2/dense_2/BiasAdd/ReadVariableOp
-sequential/time_distributed_2/dense_2/BiasAddBiasAdd6sequential/time_distributed_2/dense_2/MatMul:product:0Dsequential/time_distributed_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-sequential/time_distributed_2/dense_2/BiasAddญ
/sequential/time_distributed_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential/time_distributed_2/Reshape_1/shape/0ค
/sequential/time_distributed_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential/time_distributed_2/Reshape_1/shape/2พ
-sequential/time_distributed_2/Reshape_1/shapePack8sequential/time_distributed_2/Reshape_1/shape/0:output:04sequential/time_distributed_2/strided_slice:output:08sequential/time_distributed_2/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2/
-sequential/time_distributed_2/Reshape_1/shape
'sequential/time_distributed_2/Reshape_1Reshape6sequential/time_distributed_2/dense_2/BiasAdd:output:06sequential/time_distributed_2/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2)
'sequential/time_distributed_2/Reshape_1ฏ
-sequential/time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2/
-sequential/time_distributed_2/Reshape_2/shape๚
'sequential/time_distributed_2/Reshape_2Reshape0sequential/time_distributed_1/Reshape_1:output:06sequential/time_distributed_2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:?????????2)
'sequential/time_distributed_2/Reshape_2ฉ
IdentityIdentity0sequential/time_distributed_2/Reshape_1:output:0^sequential/lstm/while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::2.
sequential/lstm/whilesequential/lstm/while:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
๒7
ๆ
while_body_7635
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceร
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeิ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemภ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	(*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpอ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMulล
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpถ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMul_1ซ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/addพ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpธ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Reluจ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Relu_1ฌ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
ทํ
ษ	
D__inference_sequential_layer_call_and_return_conditional_losses_6549

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource6
2seq_self_attention_shape_3_readvariableop_resource6
2seq_self_attention_shape_5_readvariableop_resource4
0seq_self_attention_add_1_readvariableop_resource6
2seq_self_attention_shape_7_readvariableop_resource4
0seq_self_attention_add_2_readvariableop_resource9
5time_distributed_dense_matmul_readvariableop_resource:
6time_distributed_dense_biasadd_readvariableop_resource=
9time_distributed_1_dense_1_matmul_readvariableop_resource>
:time_distributed_1_dense_1_biasadd_readvariableop_resource=
9time_distributed_2_dense_2_matmul_readvariableop_resource>
:time_distributed_2_dense_2_biasadd_readvariableop_resource
identityข
lstm/while
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dimซ
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/conv1d/ExpandDimsฮ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimิ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:2
conv1d/conv1d/ExpandDims_1ิ
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/conv1dจ
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeezeข
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1d/BiasAdd/ReadVariableOpฉ
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2
conv1d/BiasAddr
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????2
conv1d/Relu
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dimล
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d_1/conv1d/ExpandDimsี
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_1/conv1dฎ
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeezeจ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOpฑ
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2
conv1d_1/BiasAddx
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:?????????2
conv1d_1/Relu
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dimว
conv1d_2/conv1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d_2/conv1d/ExpandDimsี
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_2/conv1dฎ
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeezeจ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOpฑ
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2
conv1d_2/BiasAddx
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????2
conv1d_2/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dimม
max_pooling1d/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
max_pooling1d/ExpandDimsส
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPoolง
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims
2
max_pooling1d/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const
flatten/ReshapeReshapemax_pooling1d/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten/Reshape~
repeat_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
repeat_vector/ExpandDims/dimบ
repeat_vector/ExpandDims
ExpandDimsflatten/Reshape:output:0%repeat_vector/ExpandDims/dim:output:0*
T0*,
_output_shapes
:?????????2
repeat_vector/ExpandDims
repeat_vector/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
repeat_vector/stackจ
repeat_vector/TileTile!repeat_vector/ExpandDims:output:0repeat_vector/stack:output:0*
T0*,
_output_shapes
:?????????2
repeat_vector/Tilec

lstm/ShapeShaperepeat_vector/Tile:output:0*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm
lstm/transpose	Transposerepeat_vector/Tile:output:0lstm/transpose/perm:output:0*
T0*,
_output_shapes
:?????????2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm/TensorArrayV2/element_shapeฦ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2ษ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
lstm/strided_slice_2ป
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOpท
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/lstm_cell/MatMulภ
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOpณ
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/lstm_cell/MatMul_1ง
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm/lstm_cell/addน
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOpด
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim๛
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm/lstm_cell/split
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Sigmoid
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Sigmoid_1
lstm/lstm_cell/mulMullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/mul
lstm/lstm_cell/ReluRelulstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Reluค
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/mul_1
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/add_1
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Sigmoid_2
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/Relu_1จ
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm/lstm_cell/mul_2
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2$
"lstm/TensorArrayV2_1/element_shapeฬ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterฌ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
lstm_while_body_6302* 
condR
lstm_while_cond_6301*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2

lstm/whileฟ
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2ธ
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permน
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimex
seq_self_attention/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape
&seq_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&seq_self_attention/strided_slice/stack
(seq_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(seq_self_attention/strided_slice/stack_1
(seq_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(seq_self_attention/strided_slice/stack_2ิ
 seq_self_attention/strided_sliceStridedSlice!seq_self_attention/Shape:output:0/seq_self_attention/strided_slice/stack:output:01seq_self_attention/strided_slice/stack_1:output:01seq_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 seq_self_attention/strided_slice|
seq_self_attention/Shape_1Shapelstm/transpose_1:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_1
(seq_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(seq_self_attention/strided_slice_1/stackข
*seq_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*seq_self_attention/strided_slice_1/stack_1ข
*seq_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*seq_self_attention/strided_slice_1/stack_2เ
"seq_self_attention/strided_slice_1StridedSlice#seq_self_attention/Shape_1:output:01seq_self_attention/strided_slice_1/stack:output:03seq_self_attention/strided_slice_1/stack_1:output:03seq_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"seq_self_attention/strided_slice_1
(seq_self_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(seq_self_attention/strided_slice_2/stackข
*seq_self_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*seq_self_attention/strided_slice_2/stack_1ข
*seq_self_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*seq_self_attention/strided_slice_2/stack_2เ
"seq_self_attention/strided_slice_2StridedSlice#seq_self_attention/Shape_1:output:01seq_self_attention/strided_slice_2/stack:output:03seq_self_attention/strided_slice_2/stack_1:output:03seq_self_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"seq_self_attention/strided_slice_2|
seq_self_attention/Shape_2Shapelstm/transpose_1:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_2
seq_self_attention/unstackUnpack#seq_self_attention/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
seq_self_attention/unstackษ
)seq_self_attention/Shape_3/ReadVariableOpReadVariableOp2seq_self_attention_shape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02+
)seq_self_attention/Shape_3/ReadVariableOp
seq_self_attention/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"
       2
seq_self_attention/Shape_3
seq_self_attention/unstack_1Unpack#seq_self_attention/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
seq_self_attention/unstack_1
 seq_self_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2"
 seq_self_attention/Reshape/shapeถ
seq_self_attention/ReshapeReshapelstm/transpose_1:y:0)seq_self_attention/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2
seq_self_attention/Reshapeอ
+seq_self_attention/transpose/ReadVariableOpReadVariableOp2seq_self_attention_shape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02-
+seq_self_attention/transpose/ReadVariableOp
!seq_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2#
!seq_self_attention/transpose/permำ
seq_self_attention/transpose	Transpose3seq_self_attention/transpose/ReadVariableOp:value:0*seq_self_attention/transpose/perm:output:0*
T0*
_output_shapes

:
 2
seq_self_attention/transpose
"seq_self_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2$
"seq_self_attention/Reshape_1/shapeฟ
seq_self_attention/Reshape_1Reshape seq_self_attention/transpose:y:0+seq_self_attention/Reshape_1/shape:output:0*
T0*
_output_shapes

:
 2
seq_self_attention/Reshape_1พ
seq_self_attention/MatMulMatMul#seq_self_attention/Reshape:output:0%seq_self_attention/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
seq_self_attention/MatMul
$seq_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_2/shape/1
$seq_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2&
$seq_self_attention/Reshape_2/shape/2
"seq_self_attention/Reshape_2/shapePack#seq_self_attention/unstack:output:0-seq_self_attention/Reshape_2/shape/1:output:0-seq_self_attention/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"seq_self_attention/Reshape_2/shapeฯ
seq_self_attention/Reshape_2Reshape#seq_self_attention/MatMul:product:0+seq_self_attention/Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
seq_self_attention/Reshape_2
!seq_self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!seq_self_attention/ExpandDims/dimู
seq_self_attention/ExpandDims
ExpandDims%seq_self_attention/Reshape_2:output:0*seq_self_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
seq_self_attention/ExpandDims|
seq_self_attention/Shape_4Shapelstm/transpose_1:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_4
seq_self_attention/unstack_2Unpack#seq_self_attention/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num2
seq_self_attention/unstack_2ษ
)seq_self_attention/Shape_5/ReadVariableOpReadVariableOp2seq_self_attention_shape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02+
)seq_self_attention/Shape_5/ReadVariableOp
seq_self_attention/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"
       2
seq_self_attention/Shape_5
seq_self_attention/unstack_3Unpack#seq_self_attention/Shape_5:output:0*
T0*
_output_shapes
: : *	
num2
seq_self_attention/unstack_3
"seq_self_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2$
"seq_self_attention/Reshape_3/shapeผ
seq_self_attention/Reshape_3Reshapelstm/transpose_1:y:0+seq_self_attention/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????
2
seq_self_attention/Reshape_3ั
-seq_self_attention/transpose_1/ReadVariableOpReadVariableOp2seq_self_attention_shape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02/
-seq_self_attention/transpose_1/ReadVariableOp
#seq_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2%
#seq_self_attention/transpose_1/perm?
seq_self_attention/transpose_1	Transpose5seq_self_attention/transpose_1/ReadVariableOp:value:0,seq_self_attention/transpose_1/perm:output:0*
T0*
_output_shapes

:
 2 
seq_self_attention/transpose_1
"seq_self_attention/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2$
"seq_self_attention/Reshape_4/shapeม
seq_self_attention/Reshape_4Reshape"seq_self_attention/transpose_1:y:0+seq_self_attention/Reshape_4/shape:output:0*
T0*
_output_shapes

:
 2
seq_self_attention/Reshape_4ฤ
seq_self_attention/MatMul_1MatMul%seq_self_attention/Reshape_3:output:0%seq_self_attention/Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2
seq_self_attention/MatMul_1
$seq_self_attention/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_5/shape/1
$seq_self_attention/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2&
$seq_self_attention/Reshape_5/shape/2
"seq_self_attention/Reshape_5/shapePack%seq_self_attention/unstack_2:output:0-seq_self_attention/Reshape_5/shape/1:output:0-seq_self_attention/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"seq_self_attention/Reshape_5/shapeั
seq_self_attention/Reshape_5Reshape%seq_self_attention/MatMul_1:product:0+seq_self_attention/Reshape_5/shape:output:0*
T0*+
_output_shapes
:????????? 2
seq_self_attention/Reshape_5
#seq_self_attention/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#seq_self_attention/ExpandDims_1/dim฿
seq_self_attention/ExpandDims_1
ExpandDims%seq_self_attention/Reshape_5:output:0,seq_self_attention/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:????????? 2!
seq_self_attention/ExpandDims_1ล
seq_self_attention/addAddV2&seq_self_attention/ExpandDims:output:0(seq_self_attention/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? 2
seq_self_attention/addฟ
'seq_self_attention/add_1/ReadVariableOpReadVariableOp0seq_self_attention_add_1_readvariableop_resource*
_output_shapes
: *
dtype02)
'seq_self_attention/add_1/ReadVariableOpฤ
seq_self_attention/add_1AddV2seq_self_attention/add:z:0/seq_self_attention/add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
seq_self_attention/add_1
seq_self_attention/TanhTanhseq_self_attention/add_1:z:0*
T0*/
_output_shapes
:????????? 2
seq_self_attention/Tanh
seq_self_attention/Shape_6Shapeseq_self_attention/Tanh:y:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_6
seq_self_attention/unstack_4Unpack#seq_self_attention/Shape_6:output:0*
T0*
_output_shapes

: : : : *	
num2
seq_self_attention/unstack_4ษ
)seq_self_attention/Shape_7/ReadVariableOpReadVariableOp2seq_self_attention_shape_7_readvariableop_resource*
_output_shapes

: *
dtype02+
)seq_self_attention/Shape_7/ReadVariableOp
seq_self_attention/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2
seq_self_attention/Shape_7
seq_self_attention/unstack_5Unpack#seq_self_attention/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2
seq_self_attention/unstack_5
"seq_self_attention/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2$
"seq_self_attention/Reshape_6/shapeร
seq_self_attention/Reshape_6Reshapeseq_self_attention/Tanh:y:0+seq_self_attention/Reshape_6/shape:output:0*
T0*'
_output_shapes
:????????? 2
seq_self_attention/Reshape_6ั
-seq_self_attention/transpose_2/ReadVariableOpReadVariableOp2seq_self_attention_shape_7_readvariableop_resource*
_output_shapes

: *
dtype02/
-seq_self_attention/transpose_2/ReadVariableOp
#seq_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2%
#seq_self_attention/transpose_2/perm?
seq_self_attention/transpose_2	Transpose5seq_self_attention/transpose_2/ReadVariableOp:value:0,seq_self_attention/transpose_2/perm:output:0*
T0*
_output_shapes

: 2 
seq_self_attention/transpose_2
"seq_self_attention/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2$
"seq_self_attention/Reshape_7/shapeม
seq_self_attention/Reshape_7Reshape"seq_self_attention/transpose_2:y:0+seq_self_attention/Reshape_7/shape:output:0*
T0*
_output_shapes

: 2
seq_self_attention/Reshape_7ฤ
seq_self_attention/MatMul_2MatMul%seq_self_attention/Reshape_6:output:0%seq_self_attention/Reshape_7:output:0*
T0*'
_output_shapes
:?????????2
seq_self_attention/MatMul_2
$seq_self_attention/Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_8/shape/1
$seq_self_attention/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_8/shape/2
$seq_self_attention/Reshape_8/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$seq_self_attention/Reshape_8/shape/3ฒ
"seq_self_attention/Reshape_8/shapePack%seq_self_attention/unstack_4:output:0-seq_self_attention/Reshape_8/shape/1:output:0-seq_self_attention/Reshape_8/shape/2:output:0-seq_self_attention/Reshape_8/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"seq_self_attention/Reshape_8/shapeี
seq_self_attention/Reshape_8Reshape%seq_self_attention/MatMul_2:product:0+seq_self_attention/Reshape_8/shape:output:0*
T0*/
_output_shapes
:?????????2
seq_self_attention/Reshape_8ฟ
'seq_self_attention/add_2/ReadVariableOpReadVariableOp0seq_self_attention_add_2_readvariableop_resource*
_output_shapes
:*
dtype02)
'seq_self_attention/add_2/ReadVariableOpฯ
seq_self_attention/add_2AddV2%seq_self_attention/Reshape_8:output:0/seq_self_attention/add_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
seq_self_attention/add_2
"seq_self_attention/Reshape_9/shapePack+seq_self_attention/strided_slice_1:output:0+seq_self_attention/strided_slice_2:output:0+seq_self_attention/strided_slice_2:output:0*
N*
T0*
_output_shapes
:2$
"seq_self_attention/Reshape_9/shapeฺ
seq_self_attention/Reshape_9Reshapeseq_self_attention/add_2:z:0+seq_self_attention/Reshape_9/shape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/Reshape_9ฉ
seq_self_attention/ReluRelu%seq_self_attention/Reshape_9:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/Relu
(seq_self_attention/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(seq_self_attention/Max/reduction_indicesแ
seq_self_attention/MaxMax%seq_self_attention/Relu:activations:01seq_self_attention/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
seq_self_attention/Maxว
seq_self_attention/subSub%seq_self_attention/Relu:activations:0seq_self_attention/Max:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/sub
seq_self_attention/ExpExpseq_self_attention/sub:z:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/Exp
(seq_self_attention/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(seq_self_attention/Sum/reduction_indicesึ
seq_self_attention/SumSumseq_self_attention/Exp:y:01seq_self_attention/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
seq_self_attention/Sumศ
seq_self_attention/truedivRealDivseq_self_attention/Exp:y:0seq_self_attention/Sum:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
seq_self_attention/truedivภ
seq_self_attention/MatMul_3BatchMatMulV2seq_self_attention/truediv:z:0lstm/transpose_1:y:0*
T0*4
_output_shapes"
 :??????????????????
2
seq_self_attention/MatMul_3
seq_self_attention/Shape_8Shape$seq_self_attention/MatMul_3:output:0*
T0*
_output_shapes
:2
seq_self_attention/Shape_8
time_distributed/ShapeShape$seq_self_attention/MatMul_3:output:0*
T0*
_output_shapes
:2
time_distributed/Shape
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$time_distributed/strided_slice/stack
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed/strided_slice/stack_1
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed/strided_slice/stack_2ศ
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
time_distributed/strided_slice
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2 
time_distributed/Reshape/shapeภ
time_distributed/ReshapeReshape$seq_self_attention/MatMul_3:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2
time_distributed/Reshapeำ
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOp5time_distributed_dense_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02.
,time_distributed/dense/MatMul/ReadVariableOpิ
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
time_distributed/dense/MatMulา
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOp6time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-time_distributed/dense/BiasAdd/ReadVariableOp?
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2 
time_distributed/dense/BiasAdd
time_distributed/dense/ReluRelu'time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
time_distributed/dense/Relu
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"time_distributed/Reshape_1/shape/0
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2$
"time_distributed/Reshape_1/shape/2?
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 time_distributed/Reshape_1/shapeู
time_distributed/Reshape_1Reshape)time_distributed/dense/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
time_distributed/Reshape_1
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2"
 time_distributed/Reshape_2/shapeฦ
time_distributed/Reshape_2Reshape$seq_self_attention/MatMul_3:output:0)time_distributed/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????
2
time_distributed/Reshape_2
time_distributed_1/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_1/Shape
&time_distributed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_1/strided_slice/stack
(time_distributed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_1/strided_slice/stack_1
(time_distributed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_1/strided_slice/stack_2ิ
 time_distributed_1/strided_sliceStridedSlice!time_distributed_1/Shape:output:0/time_distributed_1/strided_slice/stack:output:01time_distributed_1/strided_slice/stack_1:output:01time_distributed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_1/strided_slice
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_1/Reshape/shapeฦ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_1/Reshapeเ
0time_distributed_1/dense_1/MatMul/ReadVariableOpReadVariableOp9time_distributed_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0time_distributed_1/dense_1/MatMul/ReadVariableOpโ
!time_distributed_1/dense_1/MatMulMatMul#time_distributed_1/Reshape:output:08time_distributed_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2#
!time_distributed_1/dense_1/MatMul?
1time_distributed_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1time_distributed_1/dense_1/BiasAdd/ReadVariableOp๎
"time_distributed_1/dense_1/BiasAddBiasAdd+time_distributed_1/dense_1/MatMul:product:09time_distributed_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2$
"time_distributed_1/dense_1/BiasAddช
time_distributed_1/dense_1/ReluRelu+time_distributed_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2!
time_distributed_1/dense_1/Relu
$time_distributed_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_1/Reshape_1/shape/0
$time_distributed_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2&
$time_distributed_1/Reshape_1/shape/2
"time_distributed_1/Reshape_1/shapePack-time_distributed_1/Reshape_1/shape/0:output:0)time_distributed_1/strided_slice:output:0-time_distributed_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_1/Reshape_1/shapeใ
time_distributed_1/Reshape_1Reshape-time_distributed_1/dense_1/Relu:activations:0+time_distributed_1/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
time_distributed_1/Reshape_1
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2$
"time_distributed_1/Reshape_2/shapeฬ
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_1/Reshape_2
time_distributed_2/ShapeShape%time_distributed_1/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_2/Shape
&time_distributed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_2/strided_slice/stack
(time_distributed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_2/strided_slice/stack_1
(time_distributed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_2/strided_slice/stack_2ิ
 time_distributed_2/strided_sliceStridedSlice!time_distributed_2/Shape:output:0/time_distributed_2/strided_slice/stack:output:01time_distributed_2/strided_slice/stack_1:output:01time_distributed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_2/strided_slice
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_2/Reshape/shapeศ
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_2/Reshape฿
0time_distributed_2/dense_2/MatMul/ReadVariableOpReadVariableOp9time_distributed_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0time_distributed_2/dense_2/MatMul/ReadVariableOpแ
!time_distributed_2/dense_2/MatMulMatMul#time_distributed_2/Reshape:output:08time_distributed_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!time_distributed_2/dense_2/MatMul?
1time_distributed_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1time_distributed_2/dense_2/BiasAdd/ReadVariableOpํ
"time_distributed_2/dense_2/BiasAddBiasAdd+time_distributed_2/dense_2/MatMul:product:09time_distributed_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"time_distributed_2/dense_2/BiasAdd
$time_distributed_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_2/Reshape_1/shape/0
$time_distributed_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$time_distributed_2/Reshape_1/shape/2
"time_distributed_2/Reshape_1/shapePack-time_distributed_2/Reshape_1/shape/0:output:0)time_distributed_2/strided_slice:output:0-time_distributed_2/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_2/Reshape_1/shapeเ
time_distributed_2/Reshape_1Reshape+time_distributed_2/dense_2/BiasAdd:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
time_distributed_2/Reshape_1
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2$
"time_distributed_2/Reshape_2/shapeฮ
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_2/Reshape_2
IdentityIdentity%time_distributed_2/Reshape_1:output:0^lstm/while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::2

lstm/while
lstm/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ก
น
while_cond_5410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_5410___redundant_placeholder02
.while_while_cond_5410___redundant_placeholder12
.while_while_cond_5410___redundant_placeholder22
.while_while_cond_5410___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
๏
ณ
1__inference_seq_self_attention_layer_call_fn_7986

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityขStatefulPartitionedCallฐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_57402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs

ฤ
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_8070

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:?????????2	
Reshapeง
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_1/MatMulฅ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpข
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
dense_1/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????:::] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ใ

)__inference_sequential_layer_call_fn_6027
conv1d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityขStatefulPartitionedCall๚
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_59842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
ญV
๔
>__inference_lstm_layer_call_and_return_conditional_losses_7239
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityขwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shapeฒ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2ฟ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape๘
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2ฌ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02!
lstm_cell/MatMul/ReadVariableOpฃ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMulฑ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/addช
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim็
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_1
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shapeธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterแ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_7154*
condR
while_cond_7153*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
whileต
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape๑
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permฎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::2
whilewhile:_ [
5
_output_shapes#
!:??????????????????
"
_user_specified_name
inputs/0
?$
๎
while_body_4500
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_4524_0
while_lstm_cell_4526_0
while_lstm_cell_4528_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_4524
while_lstm_cell_4526
while_lstm_cell_4528ข'while/lstm_cell/StatefulPartitionedCallร
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeิ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemว
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4524_0while_lstm_cell_4526_0while_lstm_cell_4528_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????
:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_41732)
'while/lstm_cell/StatefulPartitionedCall๔
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2ท
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3พ
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2
while/Identity_4พ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0".
while_lstm_cell_4524while_lstm_cell_4524_0".
while_lstm_cell_4526while_lstm_cell_4526_0".
while_lstm_cell_4528while_lstm_cell_4528_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
ก
น
while_cond_7634
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_7634___redundant_placeholder02
.while_while_cond_7634___redundant_placeholder12
.while_while_cond_7634___redundant_placeholder22
.while_while_cond_7634___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
ฅ
ท
B__inference_conv1d_2_layer_call_and_return_conditional_losses_5159

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d/ExpandDimsบ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimน
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1ธ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????:::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ก
น
while_cond_4631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_4631___redundant_placeholder02
.while_while_cond_4631___redundant_placeholder12
.while_while_cond_4631___redundant_placeholder22
.while_while_cond_4631___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
๋
พ
J__inference_time_distributed_layer_call_and_return_conditional_losses_8030

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????
2	
Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2

dense/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapedense/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????
:::\ X
4
_output_shapes"
 :??????????????????

 
_user_specified_nameinputs
ญ

#__inference_lstm_layer_call_fn_7414
inputs_0
unknown
	unknown_0
	unknown_1
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_47012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:??????????????????
"
_user_specified_name
inputs/0
๏
ณ
1__inference_seq_self_attention_layer_call_fn_7971

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityขStatefulPartitionedCallฐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_56332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
อ
ฉ
A__inference_dense_2_layer_call_and_return_conditional_losses_8320

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


/__inference_time_distributed_layer_call_fn_8048

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_48252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????
::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????

 
_user_specified_nameinputs
์
|
'__inference_conv1d_2_layer_call_fn_7075

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๗
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_51592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ห
c
G__inference_repeat_vector_layer_call_and_return_conditional_losses_4094

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"         2
stackx
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :??????????????????2
Tilen
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
ฏ
ฉ
A__inference_dense_1_layer_call_and_return_conditional_losses_4854

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ฤ
H
,__inference_repeat_vector_layer_call_fn_4100

inputs
identityา
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_repeat_vector_layer_call_and_return_conditional_losses_40942
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????????????:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
ณ
ศ
(__inference_lstm_cell_layer_call_fn_8270

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ขStatefulPartitionedCallพ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????
:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_42062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:?????????:?????????
:?????????
:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/1
D
ล
>__inference_lstm_layer_call_and_return_conditional_losses_4701

inputs
lstm_cell_4619
lstm_cell_4621
lstm_cell_4623
identityข!lstm_cell/StatefulPartitionedCallขwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shapeฒ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2ฟ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape๘
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4619lstm_cell_4621lstm_cell_4623*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????
:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_42062#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shapeธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4619lstm_cell_4621lstm_cell_4623*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_4632*
condR
while_cond_4631*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
whileต
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape๑
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permฎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
๐
ฉ
J__inference_time_distributed_layer_call_and_return_conditional_losses_4796

inputs

dense_4786

dense_4788
identityขdense/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????
2	
Reshape
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0
dense_4786
dense_4788*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_47322
dense/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeฃ
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
	Reshape_1
IdentityIdentityReshape_1:output:0^dense/StatefulPartitionedCall*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????
::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????

 
_user_specified_nameinputs
?

C__inference_lstm_cell_layer_call_and_return_conditional_losses_4206

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimฟ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????
2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????
2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????
2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????
2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????
2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????
2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????
2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????
2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????
2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????
2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:?????????:?????????
:?????????
::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_namestates:OK
'
_output_shapes
:?????????

 
_user_specified_namestates
ก
น
while_cond_5257
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_5257___redundant_placeholder02
.while_while_cond_5257___redundant_placeholder12
.while_while_cond_5257___redundant_placeholder22
.while_while_cond_5257___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
ก
น
while_cond_7153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_7153___redundant_placeholder02
.while_while_cond_7153___redundant_placeholder12
.while_while_cond_7153___redundant_placeholder22
.while_while_cond_7153___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
ฅ
ท
B__inference_conv1d_1_layer_call_and_return_conditional_losses_7041

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d/ExpandDimsบ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimน
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1ธ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????:::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ั

)__inference_sequential_layer_call_fn_6955

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityขStatefulPartitionedCall๔
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_59842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ธ	

lstm_while_cond_6662&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1<
8lstm_while_lstm_while_cond_6662___redundant_placeholder0<
8lstm_while_lstm_while_cond_6662___redundant_placeholder1<
8lstm_while_lstm_while_cond_6662___redundant_placeholder2<
8lstm_while_lstm_while_cond_6662___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
๎U
๒
>__inference_lstm_layer_call_and_return_conditional_losses_5343

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityขwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shapeฒ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2ฟ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape๘
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2ฌ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02!
lstm_cell/MatMul/ReadVariableOpฃ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMulฑ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/addช
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim็
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_1
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shapeธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterแ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5258*
condR
while_cond_5257*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
whileต
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape่
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permฅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::2
whilewhile:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ๅ

C__inference_lstm_cell_layer_call_and_return_conditional_losses_8236

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimฟ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????
2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????
2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????
2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????
2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????
2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????
2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????
2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????
2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????
2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????
2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:?????????:?????????
:?????????
::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/1
ฅ
ท
B__inference_conv1d_2_layer_call_and_return_conditional_losses_7066

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d/ExpandDimsบ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimน
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1ธ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????:::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
พL
ถ

sequential_lstm_while_body_3823<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3;
7sequential_lstm_while_sequential_lstm_strided_slice_1_0w
ssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0D
@sequential_lstm_while_lstm_cell_matmul_readvariableop_resource_0F
Bsequential_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0E
Asequential_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"
sequential_lstm_while_identity$
 sequential_lstm_while_identity_1$
 sequential_lstm_while_identity_2$
 sequential_lstm_while_identity_3$
 sequential_lstm_while_identity_4$
 sequential_lstm_while_identity_59
5sequential_lstm_while_sequential_lstm_strided_slice_1u
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorB
>sequential_lstm_while_lstm_cell_matmul_readvariableop_resourceD
@sequential_lstm_while_lstm_cell_matmul_1_readvariableop_resourceC
?sequential_lstm_while_lstm_cell_biasadd_readvariableop_resourceใ
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2I
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeด
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0!sequential_lstm_while_placeholderPsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02;
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItem๐
5sequential/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@sequential_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	(*
dtype027
5sequential/lstm/while/lstm_cell/MatMul/ReadVariableOp
&sequential/lstm/while/lstm_cell/MatMulMatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=sequential/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2(
&sequential/lstm/while/lstm_cell/MatMul๕
7sequential/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBsequential_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype029
7sequential/lstm/while/lstm_cell/MatMul_1/ReadVariableOp๖
(sequential/lstm/while/lstm_cell/MatMul_1MatMul#sequential_lstm_while_placeholder_2?sequential/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2*
(sequential/lstm/while/lstm_cell/MatMul_1๋
#sequential/lstm/while/lstm_cell/addAddV20sequential/lstm/while/lstm_cell/MatMul:product:02sequential/lstm/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2%
#sequential/lstm/while/lstm_cell/add๎
6sequential/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAsequential_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype028
6sequential/lstm/while/lstm_cell/BiasAdd/ReadVariableOp๘
'sequential/lstm/while/lstm_cell/BiasAddBiasAdd'sequential/lstm/while/lstm_cell/add:z:0>sequential/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'sequential/lstm/while/lstm_cell/BiasAdd
%sequential/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/lstm/while/lstm_cell/Constค
/sequential/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential/lstm/while/lstm_cell/split/split_dimฟ
%sequential/lstm/while/lstm_cell/splitSplit8sequential/lstm/while/lstm_cell/split/split_dim:output:00sequential/lstm/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2'
%sequential/lstm/while/lstm_cell/splitฟ
'sequential/lstm/while/lstm_cell/SigmoidSigmoid.sequential/lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2)
'sequential/lstm/while/lstm_cell/Sigmoidร
)sequential/lstm/while/lstm_cell/Sigmoid_1Sigmoid.sequential/lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2+
)sequential/lstm/while/lstm_cell/Sigmoid_1ื
#sequential/lstm/while/lstm_cell/mulMul-sequential/lstm/while/lstm_cell/Sigmoid_1:y:0#sequential_lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????
2%
#sequential/lstm/while/lstm_cell/mulถ
$sequential/lstm/while/lstm_cell/ReluRelu.sequential/lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2&
$sequential/lstm/while/lstm_cell/Relu่
%sequential/lstm/while/lstm_cell/mul_1Mul+sequential/lstm/while/lstm_cell/Sigmoid:y:02sequential/lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2'
%sequential/lstm/while/lstm_cell/mul_1?
%sequential/lstm/while/lstm_cell/add_1AddV2'sequential/lstm/while/lstm_cell/mul:z:0)sequential/lstm/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2'
%sequential/lstm/while/lstm_cell/add_1ร
)sequential/lstm/while/lstm_cell/Sigmoid_2Sigmoid.sequential/lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2+
)sequential/lstm/while/lstm_cell/Sigmoid_2ต
&sequential/lstm/while/lstm_cell/Relu_1Relu)sequential/lstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2(
&sequential/lstm/while/lstm_cell/Relu_1์
%sequential/lstm/while/lstm_cell/mul_2Mul-sequential/lstm/while/lstm_cell/Sigmoid_2:y:04sequential/lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2'
%sequential/lstm/while/lstm_cell/mul_2ญ
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#sequential_lstm_while_placeholder_1!sequential_lstm_while_placeholder)sequential/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItem|
sequential/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/while/add/yฉ
sequential/lstm/while/addAddV2!sequential_lstm_while_placeholder$sequential/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/while/add
sequential/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/while/add_1/yฦ
sequential/lstm/while/add_1AddV28sequential_lstm_while_sequential_lstm_while_loop_counter&sequential/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/while/add_1
sequential/lstm/while/IdentityIdentitysequential/lstm/while/add_1:z:0*
T0*
_output_shapes
: 2 
sequential/lstm/while/Identityฑ
 sequential/lstm/while/Identity_1Identity>sequential_lstm_while_sequential_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_1
 sequential/lstm/while/Identity_2Identitysequential/lstm/while/add:z:0*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_2ฝ
 sequential/lstm/while/Identity_3IdentityJsequential/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_3ญ
 sequential/lstm/while/Identity_4Identity)sequential/lstm/while/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:?????????
2"
 sequential/lstm/while/Identity_4ญ
 sequential/lstm/while/Identity_5Identity)sequential/lstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2"
 sequential/lstm/while/Identity_5"I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0"M
 sequential_lstm_while_identity_1)sequential/lstm/while/Identity_1:output:0"M
 sequential_lstm_while_identity_2)sequential/lstm/while/Identity_2:output:0"M
 sequential_lstm_while_identity_3)sequential/lstm/while/Identity_3:output:0"M
 sequential_lstm_while_identity_4)sequential/lstm/while/Identity_4:output:0"M
 sequential_lstm_while_identity_5)sequential/lstm/while/Identity_5:output:0"
?sequential_lstm_while_lstm_cell_biasadd_readvariableop_resourceAsequential_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"
@sequential_lstm_while_lstm_cell_matmul_1_readvariableop_resourceBsequential_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"
>sequential_lstm_while_lstm_cell_matmul_readvariableop_resource@sequential_lstm_while_lstm_cell_matmul_readvariableop_resource_0"p
5sequential_lstm_while_sequential_lstm_strided_slice_17sequential_lstm_while_sequential_lstm_strided_slice_1_0"่
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 

ฤ
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_8131

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:?????????2	
Reshapeฆ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulReshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMulค
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpก
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapedense_2/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_1s
IdentityIdentityReshape_1:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????:::] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
๐
ฉ
J__inference_time_distributed_layer_call_and_return_conditional_losses_4825

inputs

dense_4815

dense_4817
identityขdense/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????
2	
Reshape
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0
dense_4815
dense_4817*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_47322
dense/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeฃ
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
	Reshape_1
IdentityIdentityReshape_1:output:0^dense/StatefulPartitionedCall*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????
::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????

 
_user_specified_nameinputs
ท

"__inference_signature_wrapper_6188
conv1d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityขStatefulPartitionedCallี
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_40702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input


#__inference_lstm_layer_call_fn_7731

inputs
unknown
	unknown_0
	unknown_1
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_53432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs

ต
@__inference_conv1d_layer_call_and_return_conditional_losses_5095

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDimsน
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimธ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:2
conv1d/ExpandDims_1ธ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs

ฑ
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_4947

inputs
dense_1_4937
dense_1_4939
identityขdense_1/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:?????????2	
Reshape
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_1_4937dense_1_4939*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_48542!
dense_1/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeฅ
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
	Reshape_1
IdentityIdentityReshape_1:output:0 ^dense_1/StatefulPartitionedCall*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ไ
c
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_4079

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDimsฑ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ธ	

lstm_while_cond_6301&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1<
8lstm_while_lstm_while_cond_6301___redundant_placeholder0<
8lstm_while_lstm_while_cond_6301___redundant_placeholder1<
8lstm_while_lstm_while_cond_6301___redundant_placeholder2<
8lstm_while_lstm_while_cond_6301___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
ฑZ
?
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_7956

inputs#
shape_3_readvariableop_resource#
shape_5_readvariableop_resource!
add_1_readvariableop_resource#
shape_7_readvariableop_resource!
add_2_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2๎
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2H
Shape_2Shapeinputs*
T0*
_output_shapes
:2	
Shape_2^
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"
       2	
Shape_3`
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????
2	
Reshape
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:
 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:
 2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2ข
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_2b
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsReshape_2:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2

ExpandDimsH
Shape_4Shapeinputs*
T0*
_output_shapes
:2	
Shape_4b
	unstack_2UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02
Shape_5/ReadVariableOpc
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"
       2	
Shape_5`
	unstack_3UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape_3/shapeu
	Reshape_3ReshapeinputsReshape_3/shape:output:0*
T0*'
_output_shapes
:?????????
2
	Reshape_3
transpose_1/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:
 2
transpose_1s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:
 2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_5/shape/2ค
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_5f
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsReshape_5:output:0ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:????????? 2
ExpandDims_1y
addAddV2ExpandDims:output:0ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? 2
add
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
add_1/ReadVariableOpx
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
add_1Y
TanhTanh	add_1:z:0*
T0*/
_output_shapes
:????????? 2
TanhJ
Shape_6ShapeTanh:y:0*
T0*
_output_shapes
:2	
Shape_6d
	unstack_4UnpackShape_6:output:0*
T0*
_output_shapes

: : : : *	
num2
	unstack_4
Shape_7/ReadVariableOpReadVariableOpshape_7_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_7/ReadVariableOpc
Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_7`
	unstack_5UnpackShape_7:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_5s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape_6/shapew
	Reshape_6ReshapeTanh:y:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:????????? 2
	Reshape_6
transpose_2/ReadVariableOpReadVariableOpshape_7_readvariableop_resource*
_output_shapes

: *
dtype02
transpose_2/ReadVariableOpu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_7/shapeu
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_7x
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2h
Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/1h
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/2h
Reshape_8/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/3ภ
Reshape_8/shapePackunstack_4:output:0Reshape_8/shape/1:output:0Reshape_8/shape/2:output:0Reshape_8/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_8/shape
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*/
_output_shapes
:?????????2
	Reshape_8
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
add_2/ReadVariableOp
add_2AddV2Reshape_8:output:0add_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
add_2ฆ
Reshape_9/shapePackstrided_slice_1:output:0strided_slice_2:output:0strided_slice_2:output:0*
N*
T0*
_output_shapes
:2
Reshape_9/shape
	Reshape_9Reshape	add_2:z:0Reshape_9/shape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Reshape_9p
ReluReluReshape_9:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
Reluy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices
MaxMaxRelu:activations:0Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Max{
subSubRelu:activations:0Max:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
subb
ExpExpsub:z:0*
T0*=
_output_shapes+
):'???????????????????????????2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Sum|
truedivRealDivExp:y:0Sum:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
truedivy
MatMul_3BatchMatMulV2truediv:z:0inputs*
T0*4
_output_shapes"
 :??????????????????
2

MatMul_3S
Shape_8ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_8r
IdentityIdentityMatMul_3:output:0*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs

ต
@__inference_conv1d_layer_call_and_return_conditional_losses_7016

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDimsน
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimธ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:2
conv1d/ExpandDims_1ธ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ช>
๗
lstm_while_body_6663&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_09
5lstm_while_lstm_cell_matmul_readvariableop_resource_0;
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor7
3lstm_while_lstm_cell_matmul_readvariableop_resource9
5lstm_while_lstm_cell_matmul_1_readvariableop_resource8
4lstm_while_lstm_cell_biasadd_readvariableop_resourceอ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape๒
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemฯ
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	(*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOpแ
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/while/lstm_cell/MatMulิ
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpส
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/while/lstm_cell/MatMul_1ฟ
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm/while/lstm_cell/addอ
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpฬ
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm/while/lstm_cell/BiasAddz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/Const
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dim
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm/while/lstm_cell/split
lstm/while/lstm_cell/SigmoidSigmoid#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/Sigmoidข
lstm/while/lstm_cell/Sigmoid_1Sigmoid#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2 
lstm/while/lstm_cell/Sigmoid_1ซ
lstm/while/lstm_cell/mulMul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/mul
lstm/while/lstm_cell/ReluRelu#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/Reluผ
lstm/while/lstm_cell/mul_1Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/mul_1ฑ
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/add_1ข
lstm/while/lstm_cell/Sigmoid_2Sigmoid#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2 
lstm/while/lstm_cell/Sigmoid_2
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/Relu_1ภ
lstm/while/lstm_cell/mul_2Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm/while/lstm_cell/mul_2๖
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1m
lstm/while/IdentityIdentitylstm/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm/while/Identity_1o
lstm/while/Identity_2Identitylstm/while/add:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity_2
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:?????????
2
lstm/while/Identity_4
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"ผ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 

ฑ
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_4918

inputs
dense_1_4908
dense_1_4910
identityขdense_1/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:?????????2	
Reshape
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_1_4908dense_1_4910*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_48542!
dense_1/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeฅ
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
	Reshape_1
IdentityIdentityReshape_1:output:0 ^dense_1/StatefulPartitionedCall*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ก
น
while_cond_7481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_7481___redundant_placeholder02
.while_while_cond_7481___redundant_placeholder12
.while_while_cond_7481___redundant_placeholder22
.while_while_cond_7481___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
ก
น
while_cond_4499
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_4499___redundant_placeholder02
.while_while_cond_4499___redundant_placeholder12
.while_while_cond_4499___redundant_placeholder22
.while_while_cond_4499___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
ุ
{
&__inference_dense_2_layer_call_fn_8329

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๑
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_49752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๒7
ๆ
while_body_5258
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceร
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeิ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemภ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	(*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpอ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMulล
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpถ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMul_1ซ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/addพ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpธ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Reluจ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Relu_1ฌ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
ิ@
๗
D__inference_sequential_layer_call_and_return_conditional_losses_5920
conv1d_input
conv1d_5862
conv1d_5864
conv1d_1_5867
conv1d_1_5869
conv1d_2_5872
conv1d_2_5874
	lstm_5880
	lstm_5882
	lstm_5884
seq_self_attention_5887
seq_self_attention_5889
seq_self_attention_5891
seq_self_attention_5893
seq_self_attention_5895
time_distributed_5898
time_distributed_5900
time_distributed_1_5905
time_distributed_1_5907
time_distributed_2_5912
time_distributed_2_5914
identityขconv1d/StatefulPartitionedCallข conv1d_1/StatefulPartitionedCallข conv1d_2/StatefulPartitionedCallขlstm/StatefulPartitionedCallข*seq_self_attention/StatefulPartitionedCallข(time_distributed/StatefulPartitionedCallข*time_distributed_1/StatefulPartitionedCallข*time_distributed_2/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_5862conv1d_5864*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_50952 
conv1d/StatefulPartitionedCallด
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_5867conv1d_1_5869*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_51272"
 conv1d_1/StatefulPartitionedCallถ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_5872conv1d_2_5874*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_51592"
 conv1d_2/StatefulPartitionedCall
max_pooling1d/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_40792
max_pooling1d/PartitionedCall๐
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_51822
flatten/PartitionedCall
repeat_vector/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_repeat_vector_layer_call_and_return_conditional_losses_40942
repeat_vector/PartitionedCallซ
lstm/StatefulPartitionedCallStatefulPartitionedCall&repeat_vector/PartitionedCall:output:0	lstm_5880	lstm_5882	lstm_5884*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_54962
lstm/StatefulPartitionedCallฝ
*seq_self_attention/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0seq_self_attention_5887seq_self_attention_5889seq_self_attention_5891seq_self_attention_5893seq_self_attention_5895*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_57402,
*seq_self_attention/StatefulPartitionedCall๑
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall3seq_self_attention/StatefulPartitionedCall:output:0time_distributed_5898time_distributed_5900*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_48252*
(time_distributed/StatefulPartitionedCall
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2 
time_distributed/Reshape/shapeฯ
time_distributed/ReshapeReshape3seq_self_attention/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2
time_distributed/Reshape๙
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_5905time_distributed_1_5907*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_49472,
*time_distributed_1/StatefulPartitionedCall
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_1/Reshape/shapeิ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_1/Reshape๚
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0time_distributed_2_5912time_distributed_2_5914*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_50682,
*time_distributed_2/StatefulPartitionedCall
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_2/Reshape/shapeึ
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_2/Reshapeฬ
IdentityIdentity3time_distributed_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall+^seq_self_attention/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2X
*seq_self_attention/StatefulPartitionedCall*seq_self_attention/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
ณ
]
A__inference_flatten_layer_call_and_return_conditional_losses_7081

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ฅ
ท
B__inference_conv1d_1_layer_call_and_return_conditional_losses_5127

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2
conv1d/ExpandDimsบ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimน
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1ธ
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????:::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
๒7
ๆ
while_body_7154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resourceร
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeิ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemภ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	(*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOpอ
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMulล
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:
(*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOpถ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/MatMul_1ซ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/addพ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
:(*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOpธ
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
while/lstm_cell/split
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Reluจ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_1
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/Relu_1ฌ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
ิ
y
$__inference_dense_layer_call_fn_8290

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall๐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_47322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
ย@
๑
D__inference_sequential_layer_call_and_return_conditional_losses_5984

inputs
conv1d_5926
conv1d_5928
conv1d_1_5931
conv1d_1_5933
conv1d_2_5936
conv1d_2_5938
	lstm_5944
	lstm_5946
	lstm_5948
seq_self_attention_5951
seq_self_attention_5953
seq_self_attention_5955
seq_self_attention_5957
seq_self_attention_5959
time_distributed_5962
time_distributed_5964
time_distributed_1_5969
time_distributed_1_5971
time_distributed_2_5976
time_distributed_2_5978
identityขconv1d/StatefulPartitionedCallข conv1d_1/StatefulPartitionedCallข conv1d_2/StatefulPartitionedCallขlstm/StatefulPartitionedCallข*seq_self_attention/StatefulPartitionedCallข(time_distributed/StatefulPartitionedCallข*time_distributed_1/StatefulPartitionedCallข*time_distributed_2/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_5926conv1d_5928*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_50952 
conv1d/StatefulPartitionedCallด
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_5931conv1d_1_5933*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_51272"
 conv1d_1/StatefulPartitionedCallถ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_5936conv1d_2_5938*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_51592"
 conv1d_2/StatefulPartitionedCall
max_pooling1d/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_40792
max_pooling1d/PartitionedCall๐
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_51822
flatten/PartitionedCall
repeat_vector/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_repeat_vector_layer_call_and_return_conditional_losses_40942
repeat_vector/PartitionedCallซ
lstm/StatefulPartitionedCallStatefulPartitionedCall&repeat_vector/PartitionedCall:output:0	lstm_5944	lstm_5946	lstm_5948*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_53432
lstm/StatefulPartitionedCallฝ
*seq_self_attention/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0seq_self_attention_5951seq_self_attention_5953seq_self_attention_5955seq_self_attention_5957seq_self_attention_5959*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_56332,
*seq_self_attention/StatefulPartitionedCall๑
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall3seq_self_attention/StatefulPartitionedCall:output:0time_distributed_5962time_distributed_5964*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_47962*
(time_distributed/StatefulPartitionedCall
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2 
time_distributed/Reshape/shapeฯ
time_distributed/ReshapeReshape3seq_self_attention/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2
time_distributed/Reshape๙
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_5969time_distributed_1_5971*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_49182,
*time_distributed_1/StatefulPartitionedCall
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_1/Reshape/shapeิ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_1/Reshape๚
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0time_distributed_2_5976time_distributed_2_5978*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_50392,
*time_distributed_2/StatefulPartitionedCall
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_2/Reshape/shapeึ
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_2/Reshapeฬ
IdentityIdentity3time_distributed_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall+^seq_self_attention/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2X
*seq_self_attention/StatefulPartitionedCall*seq_self_attention/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ช
ง
?__inference_dense_layer_call_and_return_conditional_losses_8281

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
ฃ

1__inference_time_distributed_2_layer_call_fn_8161

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_50392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ฑZ
?
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_5633

inputs#
shape_3_readvariableop_resource#
shape_5_readvariableop_resource!
add_1_readvariableop_resource#
shape_7_readvariableop_resource!
add_2_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2๎
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2H
Shape_2Shapeinputs*
T0*
_output_shapes
:2	
Shape_2^
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"
       2	
Shape_3`
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????
2	
Reshape
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:
 *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:
 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:
 2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2ข
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_2b
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsReshape_2:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2

ExpandDimsH
Shape_4Shapeinputs*
T0*
_output_shapes
:2	
Shape_4b
	unstack_2UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02
Shape_5/ReadVariableOpc
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"
       2	
Shape_5`
	unstack_3UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Reshape_3/shapeu
	Reshape_3ReshapeinputsReshape_3/shape:output:0*
T0*'
_output_shapes
:?????????
2
	Reshape_3
transpose_1/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:
 *
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:
 2
transpose_1s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   ????2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:
 2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_5/shape/2ค
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:????????? 2
	Reshape_5f
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsReshape_5:output:0ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:????????? 2
ExpandDims_1y
addAddV2ExpandDims:output:0ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? 2
add
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
add_1/ReadVariableOpx
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
add_1Y
TanhTanh	add_1:z:0*
T0*/
_output_shapes
:????????? 2
TanhJ
Shape_6ShapeTanh:y:0*
T0*
_output_shapes
:2	
Shape_6d
	unstack_4UnpackShape_6:output:0*
T0*
_output_shapes

: : : : *	
num2
	unstack_4
Shape_7/ReadVariableOpReadVariableOpshape_7_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_7/ReadVariableOpc
Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_7`
	unstack_5UnpackShape_7:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_5s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape_6/shapew
	Reshape_6ReshapeTanh:y:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:????????? 2
	Reshape_6
transpose_2/ReadVariableOpReadVariableOpshape_7_readvariableop_resource*
_output_shapes

: *
dtype02
transpose_2/ReadVariableOpu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_7/shapeu
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_7x
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2h
Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/1h
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/2h
Reshape_8/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/3ภ
Reshape_8/shapePackunstack_4:output:0Reshape_8/shape/1:output:0Reshape_8/shape/2:output:0Reshape_8/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_8/shape
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*/
_output_shapes
:?????????2
	Reshape_8
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype02
add_2/ReadVariableOp
add_2AddV2Reshape_8:output:0add_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
add_2ฆ
Reshape_9/shapePackstrided_slice_1:output:0strided_slice_2:output:0strided_slice_2:output:0*
N*
T0*
_output_shapes
:2
Reshape_9/shape
	Reshape_9Reshape	add_2:z:0Reshape_9/shape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Reshape_9p
ReluReluReshape_9:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
Reluy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices
MaxMaxRelu:activations:0Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Max{
subSubRelu:activations:0Max:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
subb
ExpExpsub:z:0*
T0*=
_output_shapes+
):'???????????????????????????2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Sum|
truedivRealDivExp:y:0Sum:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
truedivy
MatMul_3BatchMatMulV2truediv:z:0inputs*
T0*4
_output_shapes"
 :??????????????????
2

MatMul_3S
Shape_8ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_8r
IdentityIdentityMatMul_3:output:0*
T0*4
_output_shapes"
 :??????????????????
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????
::::::S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
๎U
๒
>__inference_lstm_layer_call_and_return_conditional_losses_5496

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identityขwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :่2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2๎
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shapeฒ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2ฟ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape๘
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2ฌ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02!
lstm_cell/MatMul/ReadVariableOpฃ
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMulฑ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:
(*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/addช
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim็
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????
:?????????
:?????????
:?????????
*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_1
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_1
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/Relu_1
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????
2
lstm_cell/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shapeธ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterแ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5411*
condR
while_cond_5410*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
whileต
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape่
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permฅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::2
whilewhile:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs

ฤ
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_8092

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:?????????2	
Reshapeง
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_1/MatMulฅ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpข
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
dense_1/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/2จ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:??????????????????2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????:::] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ิ@
๗
D__inference_sequential_layer_call_and_return_conditional_losses_5859
conv1d_input
conv1d_5106
conv1d_5108
conv1d_1_5138
conv1d_1_5140
conv1d_2_5170
conv1d_2_5172
	lstm_5519
	lstm_5521
	lstm_5523
seq_self_attention_5772
seq_self_attention_5774
seq_self_attention_5776
seq_self_attention_5778
seq_self_attention_5780
time_distributed_5801
time_distributed_5803
time_distributed_1_5826
time_distributed_1_5828
time_distributed_2_5851
time_distributed_2_5853
identityขconv1d/StatefulPartitionedCallข conv1d_1/StatefulPartitionedCallข conv1d_2/StatefulPartitionedCallขlstm/StatefulPartitionedCallข*seq_self_attention/StatefulPartitionedCallข(time_distributed/StatefulPartitionedCallข*time_distributed_1/StatefulPartitionedCallข*time_distributed_2/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_5106conv1d_5108*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_50952 
conv1d/StatefulPartitionedCallด
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_5138conv1d_1_5140*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_51272"
 conv1d_1/StatefulPartitionedCallถ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_5170conv1d_2_5172*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_51592"
 conv1d_2/StatefulPartitionedCall
max_pooling1d/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_40792
max_pooling1d/PartitionedCall๐
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_51822
flatten/PartitionedCall
repeat_vector/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_repeat_vector_layer_call_and_return_conditional_losses_40942
repeat_vector/PartitionedCallซ
lstm/StatefulPartitionedCallStatefulPartitionedCall&repeat_vector/PartitionedCall:output:0	lstm_5519	lstm_5521	lstm_5523*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_53432
lstm/StatefulPartitionedCallฝ
*seq_self_attention/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0seq_self_attention_5772seq_self_attention_5774seq_self_attention_5776seq_self_attention_5778seq_self_attention_5780*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_56332,
*seq_self_attention/StatefulPartitionedCall๑
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall3seq_self_attention/StatefulPartitionedCall:output:0time_distributed_5801time_distributed_5803*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_47962*
(time_distributed/StatefulPartitionedCall
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2 
time_distributed/Reshape/shapeฯ
time_distributed/ReshapeReshape3seq_self_attention/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
2
time_distributed/Reshape๙
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_5826time_distributed_1_5828*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_49182,
*time_distributed_1/StatefulPartitionedCall
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_1/Reshape/shapeิ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_1/Reshape๚
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0time_distributed_2_5851time_distributed_2_5853*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_50392,
*time_distributed_2/StatefulPartitionedCall
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2"
 time_distributed_2/Reshape/shapeึ
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*(
_output_shapes
:?????????2
time_distributed_2/Reshapeฬ
IdentityIdentity3time_distributed_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall+^seq_self_attention/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2X
*seq_self_attention/StatefulPartitionedCall*seq_self_attention/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_nameconv1d_input
ฅ

1__inference_time_distributed_1_layer_call_fn_8101

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_49182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
?$
๎
while_body_4632
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_4656_0
while_lstm_cell_4658_0
while_lstm_cell_4660_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_4656
while_lstm_cell_4658
while_lstm_cell_4660ข'while/lstm_cell/StatefulPartitionedCallร
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeิ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemว
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4656_0while_lstm_cell_4658_0while_lstm_cell_4660_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????
:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_42062)
'while/lstm_cell/StatefulPartitionedCall๔
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2ท
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3พ
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2
while/Identity_4พ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0".
while_lstm_cell_4656while_lstm_cell_4656_0".
while_lstm_cell_4658while_lstm_cell_4658_0".
while_lstm_cell_4660while_lstm_cell_4660_0"0
while_strided_slice_1while_strided_slice_1_0"จ
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
ฃ

1__inference_time_distributed_2_layer_call_fn_8170

inputs
unknown
	unknown_0
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_50682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs"ธL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ะ
serving_defaultผ
I
conv1d_input9
serving_default_conv1d_input:0?????????S
time_distributed_2=
StatefulPartitionedCall:0??????????????????tensorflow/serving/predict:ั๎
ฮg
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+๊&call_and_return_all_conditional_losses
๋_default_save_signature
์__call__"?c
_tf_keras_sequentialc{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector", "trainable": true, "dtype": "float32", "n": 1}}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 10, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "SeqSelfAttention", "config": {"name": "seq_self_attention", "trainable": true, "dtype": "float32", "units": 32, "attention_width": null, "attention_type": "additive", "return_attention": false, "history_only": false, "use_additive_bias": true, "use_attention_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "attention_activation": "relu", "attention_regularizer_weight": 0.0}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector", "trainable": true, "dtype": "float32", "n": 1}}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 10, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "SeqSelfAttention", "config": {"name": "seq_self_attention", "trainable": true, "dtype": "float32", "units": 32, "attention_width": null, "attention_type": "additive", "return_attention": false, "history_only": false, "use_additive_bias": true, "use_attention_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "attention_activation": "relu", "attention_regularizer_weight": 0.0}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 2.499999936844688e-06, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ู


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+ํ&call_and_return_all_conditional_losses
๎__call__"ฒ	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 1]}}
๊	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+๏&call_and_return_all_conditional_losses
๐__call__"ร
_tf_keras_layerฉ{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 128]}}
๊	

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
+๑&call_and_return_all_conditional_losses
๒__call__"ร
_tf_keras_layerฉ{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 256]}}
๗
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+๓&call_and_return_all_conditional_losses
๔__call__"ๆ
_tf_keras_layerฬ{"class_name": "MaxPooling1D", "name": "max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ไ
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+๕&call_and_return_all_conditional_losses
๖__call__"ำ
_tf_keras_layerน{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+๗&call_and_return_all_conditional_losses
๘__call__"ฬ
_tf_keras_layerฒ{"class_name": "RepeatVector", "name": "repeat_vector", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "repeat_vector", "trainable": true, "dtype": "float32", "n": 1}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ฝ
0cell
1
state_spec
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+๙&call_and_return_all_conditional_losses
๚__call__"

_tf_keras_rnn_layer๔	{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 10, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1536]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1536]}}
ศ	
6seq_self_attention_Add_Wt
6Wt
7seq_self_attention_Add_Wx
7Wx
8seq_self_attention_Add_bh
8bh
9seq_self_attention_Add_Wa
9Wa
:seq_self_attention_Add_ba
:ba
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+๛&call_and_return_all_conditional_losses
?__call__"๔
_tf_keras_layerฺ{"class_name": "SeqSelfAttention", "name": "seq_self_attention", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "seq_self_attention", "trainable": true, "dtype": "float32", "units": 32, "attention_width": null, "attention_type": "additive", "return_attention": false, "history_only": false, "use_additive_bias": true, "use_attention_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "attention_activation": "relu", "attention_regularizer_weight": 0.0}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 10]}}
๓
	?layer
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"ื
_tf_keras_layerฝ{"class_name": "TimeDistributed", "name": "time_distributed", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 10], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 10]}}
๛
	Dlayer
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
__call__"฿
_tf_keras_layerล{"class_name": "TimeDistributed", "name": "time_distributed_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 400], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 400]}}
๛
	Ilayer
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
+&call_and_return_all_conditional_losses
__call__"฿
_tf_keras_layerล{"class_name": "TimeDistributed", "name": "time_distributed_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 400], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 400]}}
ใ
Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_ratemยmรmฤmลmฦmว6mศ7mษ8mส9mห:mฬSmอTmฮUmฯVmะWmัXmาYmำZmิ[mีvึvืvุvูvฺv?6v?7v?8v?9v฿:vเSvแTvโUvใVvไWvๅXvๆYv็Zv่[v้"
	optimizer
 "
trackable_list_wrapper
ถ
0
1
2
3
4
5
S6
T7
U8
69
710
811
912
:13
V14
W15
X16
Y17
Z18
[19"
trackable_list_wrapper
ถ
0
1
2
3
4
5
S6
T7
U8
69
710
811
912
:13
V14
W15
X16
Y17
Z18
[19"
trackable_list_wrapper
ฮ
\layer_regularization_losses
]non_trainable_variables
regularization_losses
trainable_variables
^layer_metrics

_layers
`metrics
	variables
์__call__
๋_default_save_signature
+๊&call_and_return_all_conditional_losses
'๊"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
$:"2conv1d/kernel
:2conv1d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
ฐ
alayer_regularization_losses
bnon_trainable_variables
trainable_variables
regularization_losses
clayer_metrics

dlayers
emetrics
	variables
๎__call__
+ํ&call_and_return_all_conditional_losses
'ํ"call_and_return_conditional_losses"
_generic_user_object
':%2conv1d_1/kernel
:2conv1d_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
ฐ
flayer_regularization_losses
gnon_trainable_variables
trainable_variables
regularization_losses
hlayer_metrics

ilayers
jmetrics
	variables
๐__call__
+๏&call_and_return_all_conditional_losses
'๏"call_and_return_conditional_losses"
_generic_user_object
':%2conv1d_2/kernel
:2conv1d_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
ฐ
klayer_regularization_losses
lnon_trainable_variables
 trainable_variables
!regularization_losses
mlayer_metrics

nlayers
ometrics
"	variables
๒__call__
+๑&call_and_return_all_conditional_losses
'๑"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฐ
player_regularization_losses
qnon_trainable_variables
$trainable_variables
%regularization_losses
rlayer_metrics

slayers
tmetrics
&	variables
๔__call__
+๓&call_and_return_all_conditional_losses
'๓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฐ
ulayer_regularization_losses
vnon_trainable_variables
(trainable_variables
)regularization_losses
wlayer_metrics

xlayers
ymetrics
*	variables
๖__call__
+๕&call_and_return_all_conditional_losses
'๕"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฐ
zlayer_regularization_losses
{non_trainable_variables
,trainable_variables
-regularization_losses
|layer_metrics

}layers
~metrics
.	variables
๘__call__
+๗&call_and_return_all_conditional_losses
'๗"call_and_return_conditional_losses"
_generic_user_object
ฉ

Skernel
Trecurrent_kernel
Ubias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"้
_tf_keras_layerฯ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
S0
T1
U2"
trackable_list_wrapper
5
S0
T1
U2"
trackable_list_wrapper
ย
states
 layer_regularization_losses
non_trainable_variables
2regularization_losses
3trainable_variables
layer_metrics
layers
metrics
4	variables
๚__call__
+๙&call_and_return_all_conditional_losses
'๙"call_and_return_conditional_losses"
_generic_user_object
>:<
 2,seq_self_attention/seq_self_attention_Add_Wt
>:<
 2,seq_self_attention/seq_self_attention_Add_Wx
::8 2,seq_self_attention/seq_self_attention_Add_bh
>:< 2,seq_self_attention/seq_self_attention_Add_Wa
::82,seq_self_attention/seq_self_attention_Add_ba
C
60
71
82
93
:4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
60
71
82
93
:4"
trackable_list_wrapper
ต
 layer_regularization_losses
non_trainable_variables
;trainable_variables
<regularization_losses
layer_metrics
layers
metrics
=	variables
?__call__
+๛&call_and_return_all_conditional_losses
'๛"call_and_return_conditional_losses"
_generic_user_object
ช

Vkernel
Wbias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"?
_tf_keras_layerๅ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}}
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
ต
 layer_regularization_losses
non_trainable_variables
@trainable_variables
Aregularization_losses
layer_metrics
layers
metrics
B	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
ฏ

Xkernel
Ybias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer๊{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}}
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
ต
 layer_regularization_losses
non_trainable_variables
Etrainable_variables
Fregularization_losses
layer_metrics
layers
metrics
G	variables
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
ฏ

Zkernel
[bias
?trainable_variables
กregularization_losses
ข	variables
ฃ	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer๊{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}}
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
ต
 คlayer_regularization_losses
ฅnon_trainable_variables
Jtrainable_variables
Kregularization_losses
ฆlayer_metrics
งlayers
จmetrics
L	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(:&	(2lstm/lstm_cell/kernel
1:/
(2lstm/lstm_cell/recurrent_kernel
!:(2lstm/lstm_cell/bias
*:(	
2time_distributed/kernel
$:"2time_distributed/bias
-:+
2time_distributed_1/kernel
&:$2time_distributed_1/bias
,:*	2time_distributed_2/kernel
%:#2time_distributed_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
(
ฉ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
S0
T1
U2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
S0
T1
U2"
trackable_list_wrapper
ท
 ชlayer_regularization_losses
ซnon_trainable_variables
trainable_variables
regularization_losses
ฌlayer_metrics
ญlayers
ฎmetrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
ธ
 ฏlayer_regularization_losses
ฐnon_trainable_variables
trainable_variables
regularization_losses
ฑlayer_metrics
ฒlayers
ณmetrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
ธ
 ดlayer_regularization_losses
ตnon_trainable_variables
trainable_variables
regularization_losses
ถlayer_metrics
ทlayers
ธmetrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
D0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
ธ
 นlayer_regularization_losses
บnon_trainable_variables
?trainable_variables
กregularization_losses
ปlayer_metrics
ผlayers
ฝmetrics
ข	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
ฟ

พtotal

ฟcount
ภ	variables
ม	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
พ0
ฟ1"
trackable_list_wrapper
.
ภ	variables"
_generic_user_object
):'2Adam/conv1d/kernel/m
:2Adam/conv1d/bias/m
,:*2Adam/conv1d_1/kernel/m
!:2Adam/conv1d_1/bias/m
,:*2Adam/conv1d_2/kernel/m
!:2Adam/conv1d_2/bias/m
C:A
 23Adam/seq_self_attention/seq_self_attention_Add_Wt/m
C:A
 23Adam/seq_self_attention/seq_self_attention_Add_Wx/m
?:= 23Adam/seq_self_attention/seq_self_attention_Add_bh/m
C:A 23Adam/seq_self_attention/seq_self_attention_Add_Wa/m
?:=23Adam/seq_self_attention/seq_self_attention_Add_ba/m
-:+	(2Adam/lstm/lstm_cell/kernel/m
6:4
(2&Adam/lstm/lstm_cell/recurrent_kernel/m
&:$(2Adam/lstm/lstm_cell/bias/m
/:-	
2Adam/time_distributed/kernel/m
):'2Adam/time_distributed/bias/m
2:0
2 Adam/time_distributed_1/kernel/m
+:)2Adam/time_distributed_1/bias/m
1:/	2 Adam/time_distributed_2/kernel/m
*:(2Adam/time_distributed_2/bias/m
):'2Adam/conv1d/kernel/v
:2Adam/conv1d/bias/v
,:*2Adam/conv1d_1/kernel/v
!:2Adam/conv1d_1/bias/v
,:*2Adam/conv1d_2/kernel/v
!:2Adam/conv1d_2/bias/v
C:A
 23Adam/seq_self_attention/seq_self_attention_Add_Wt/v
C:A
 23Adam/seq_self_attention/seq_self_attention_Add_Wx/v
?:= 23Adam/seq_self_attention/seq_self_attention_Add_bh/v
C:A 23Adam/seq_self_attention/seq_self_attention_Add_Wa/v
?:=23Adam/seq_self_attention/seq_self_attention_Add_ba/v
-:+	(2Adam/lstm/lstm_cell/kernel/v
6:4
(2&Adam/lstm/lstm_cell/recurrent_kernel/v
&:$(2Adam/lstm/lstm_cell/bias/v
/:-	
2Adam/time_distributed/kernel/v
):'2Adam/time_distributed/bias/v
2:0
2 Adam/time_distributed_1/kernel/v
+:)2Adam/time_distributed_1/bias/v
1:/	2 Adam/time_distributed_2/kernel/v
*:(2Adam/time_distributed_2/bias/v
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_5859
D__inference_sequential_layer_call_and_return_conditional_losses_6910
D__inference_sequential_layer_call_and_return_conditional_losses_6549
D__inference_sequential_layer_call_and_return_conditional_losses_5920ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ๆ2ใ
__inference__wrapped_model_4070ฟ
ฒ
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช */ข,
*'
conv1d_input?????????
๒2๏
)__inference_sequential_layer_call_fn_6027
)__inference_sequential_layer_call_fn_6133
)__inference_sequential_layer_call_fn_6955
)__inference_sequential_layer_call_fn_7000ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
๊2็
@__inference_conv1d_layer_call_and_return_conditional_losses_7016ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ฯ2ฬ
%__inference_conv1d_layer_call_fn_7025ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv1d_1_layer_call_and_return_conditional_losses_7041ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_conv1d_1_layer_call_fn_7050ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv1d_2_layer_call_and_return_conditional_losses_7066ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_conv1d_2_layer_call_fn_7075ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ข2
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_4079ำ
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *3ข0
.+'???????????????????????????
2
,__inference_max_pooling1d_layer_call_fn_4085ำ
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *3ข0
.+'???????????????????????????
๋2่
A__inference_flatten_layer_call_and_return_conditional_losses_7081ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ะ2อ
&__inference_flatten_layer_call_fn_7086ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
G__inference_repeat_vector_layer_call_and_return_conditional_losses_4094ฦ
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *&ข#
!??????????????????
๚2๗
,__inference_repeat_vector_layer_call_fn_4100ฦ
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *&ข#
!??????????????????
?2ุ
>__inference_lstm_layer_call_and_return_conditional_losses_7392
>__inference_lstm_layer_call_and_return_conditional_losses_7239
>__inference_lstm_layer_call_and_return_conditional_losses_7567
>__inference_lstm_layer_call_and_return_conditional_losses_7720ี
ฬฒศ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
๏2์
#__inference_lstm_layer_call_fn_7742
#__inference_lstm_layer_call_fn_7414
#__inference_lstm_layer_call_fn_7731
#__inference_lstm_layer_call_fn_7403ี
ฬฒศ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
๔2๑
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_7956
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_7849า
ษฒล
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkwjkwargs
defaults

 

kwonlyargs

jtraining%
kwonlydefaultsช

trainingp 
annotationsช *
 
พ2ป
1__inference_seq_self_attention_layer_call_fn_7971
1__inference_seq_self_attention_layer_call_fn_7986า
ษฒล
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkwjkwargs
defaults

 

kwonlyargs

jtraining%
kwonlydefaultsช

trainingp 
annotationsช *
 
?2?
J__inference_time_distributed_layer_call_and_return_conditional_losses_8030
J__inference_time_distributed_layer_call_and_return_conditional_losses_8008ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
จ2ฅ
/__inference_time_distributed_layer_call_fn_8039
/__inference_time_distributed_layer_call_fn_8048ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
โ2฿
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_8092
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_8070ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ฌ2ฉ
1__inference_time_distributed_1_layer_call_fn_8110
1__inference_time_distributed_1_layer_call_fn_8101ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
โ2฿
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_8152
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_8131ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ฌ2ฉ
1__inference_time_distributed_2_layer_call_fn_8161
1__inference_time_distributed_2_layer_call_fn_8170ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
6B4
"__inference_signature_wrapper_6188conv1d_input
ฮ2ห
C__inference_lstm_cell_layer_call_and_return_conditional_losses_8236
C__inference_lstm_cell_layer_call_and_return_conditional_losses_8203พ
ตฒฑ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
2
(__inference_lstm_cell_layer_call_fn_8270
(__inference_lstm_cell_layer_call_fn_8253พ
ตฒฑ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
้2ๆ
?__inference_dense_layer_call_and_return_conditional_losses_8281ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ฮ2ห
$__inference_dense_layer_call_fn_8290ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_dense_1_layer_call_and_return_conditional_losses_8301ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ะ2อ
&__inference_dense_1_layer_call_fn_8310ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_dense_2_layer_call_and_return_conditional_losses_8320ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ะ2อ
&__inference_dense_2_layer_call_fn_8329ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ห
__inference__wrapped_model_4070งSTU6789:VWXYZ[9ข6
/ข,
*'
conv1d_input?????????
ช "TชQ
O
time_distributed_296
time_distributed_2??????????????????ฌ
B__inference_conv1d_1_layer_call_and_return_conditional_losses_7041f4ข1
*ข'
%"
inputs?????????
ช "*ข'
 
0?????????
 
'__inference_conv1d_1_layer_call_fn_7050Y4ข1
*ข'
%"
inputs?????????
ช "?????????ฌ
B__inference_conv1d_2_layer_call_and_return_conditional_losses_7066f4ข1
*ข'
%"
inputs?????????
ช "*ข'
 
0?????????
 
'__inference_conv1d_2_layer_call_fn_7075Y4ข1
*ข'
%"
inputs?????????
ช "?????????ฉ
@__inference_conv1d_layer_call_and_return_conditional_losses_7016e3ข0
)ข&
$!
inputs?????????
ช "*ข'
 
0?????????
 
%__inference_conv1d_layer_call_fn_7025X3ข0
)ข&
$!
inputs?????????
ช "?????????ฃ
A__inference_dense_1_layer_call_and_return_conditional_losses_8301^XY0ข-
&ข#
!
inputs?????????
ช "&ข#

0?????????
 {
&__inference_dense_1_layer_call_fn_8310QXY0ข-
&ข#
!
inputs?????????
ช "?????????ข
A__inference_dense_2_layer_call_and_return_conditional_losses_8320]Z[0ข-
&ข#
!
inputs?????????
ช "%ข"

0?????????
 z
&__inference_dense_2_layer_call_fn_8329PZ[0ข-
&ข#
!
inputs?????????
ช "??????????
?__inference_dense_layer_call_and_return_conditional_losses_8281]VW/ข,
%ข"
 
inputs?????????

ช "&ข#

0?????????
 x
$__inference_dense_layer_call_fn_8290PVW/ข,
%ข"
 
inputs?????????

ช "?????????ฃ
A__inference_flatten_layer_call_and_return_conditional_losses_7081^4ข1
*ข'
%"
inputs?????????
ช "&ข#

0?????????
 {
&__inference_flatten_layer_call_fn_7086Q4ข1
*ข'
%"
inputs?????????
ช "?????????ฦ
C__inference_lstm_cell_layer_call_and_return_conditional_losses_8203?STUข~
wขt
!
inputs?????????
KขH
"
states/0?????????

"
states/1?????????

p
ช "sขp
iขf

0/0?????????

EB

0/1/0?????????


0/1/1?????????

 ฦ
C__inference_lstm_cell_layer_call_and_return_conditional_losses_8236?STUข~
wขt
!
inputs?????????
KขH
"
states/0?????????

"
states/1?????????

p 
ช "sขp
iขf

0/0?????????

EB

0/1/0?????????


0/1/1?????????

 
(__inference_lstm_cell_layer_call_fn_8253๎STUข~
wขt
!
inputs?????????
KขH
"
states/0?????????

"
states/1?????????

p
ช "cข`

0?????????

A>

1/0?????????


1/1?????????

(__inference_lstm_cell_layer_call_fn_8270๎STUข~
wขt
!
inputs?????????
KขH
"
states/0?????????

"
states/1?????????

p 
ช "cข`

0?????????

A>

1/0?????????


1/1?????????
ฮ
>__inference_lstm_layer_call_and_return_conditional_losses_7239STUPขM
FขC
52
0-
inputs/0??????????????????

 
p

 
ช "2ข/
(%
0??????????????????

 ฮ
>__inference_lstm_layer_call_and_return_conditional_losses_7392STUPขM
FขC
52
0-
inputs/0??????????????????

 
p 

 
ช "2ข/
(%
0??????????????????

 ด
>__inference_lstm_layer_call_and_return_conditional_losses_7567rSTU@ข=
6ข3
%"
inputs?????????

 
p

 
ช ")ข&

0?????????

 ด
>__inference_lstm_layer_call_and_return_conditional_losses_7720rSTU@ข=
6ข3
%"
inputs?????????

 
p 

 
ช ")ข&

0?????????

 ฅ
#__inference_lstm_layer_call_fn_7403~STUPขM
FขC
52
0-
inputs/0??????????????????

 
p

 
ช "%"??????????????????
ฅ
#__inference_lstm_layer_call_fn_7414~STUPขM
FขC
52
0-
inputs/0??????????????????

 
p 

 
ช "%"??????????????????

#__inference_lstm_layer_call_fn_7731eSTU@ข=
6ข3
%"
inputs?????????

 
p

 
ช "?????????

#__inference_lstm_layer_call_fn_7742eSTU@ข=
6ข3
%"
inputs?????????

 
p 

 
ช "?????????
ะ
G__inference_max_pooling1d_layer_call_and_return_conditional_losses_4079EขB
;ข8
63
inputs'???????????????????????????
ช ";ข8
1.
0'???????????????????????????
 ง
,__inference_max_pooling1d_layer_call_fn_4085wEขB
;ข8
63
inputs'???????????????????????????
ช ".+'???????????????????????????น
G__inference_repeat_vector_layer_call_and_return_conditional_losses_4094n8ข5
.ข+
)&
inputs??????????????????
ช "2ข/
(%
0??????????????????
 
,__inference_repeat_vector_layer_call_fn_4100a8ข5
.ข+
)&
inputs??????????????????
ช "%"??????????????????ี
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_78496789:GขD
-ข*
$!
inputs?????????


 
ช

trainingp"2ข/
(%
0??????????????????

 ี
L__inference_seq_self_attention_layer_call_and_return_conditional_losses_79566789:GขD
-ข*
$!
inputs?????????


 
ช

trainingp "2ข/
(%
0??????????????????

 ฌ
1__inference_seq_self_attention_layer_call_fn_7971w6789:GขD
-ข*
$!
inputs?????????


 
ช

trainingp"%"??????????????????
ฌ
1__inference_seq_self_attention_layer_call_fn_7986w6789:GขD
-ข*
$!
inputs?????????


 
ช

trainingp "%"??????????????????
ึ
D__inference_sequential_layer_call_and_return_conditional_losses_5859STU6789:VWXYZ[Aข>
7ข4
*'
conv1d_input?????????
p

 
ช "2ข/
(%
0??????????????????
 ึ
D__inference_sequential_layer_call_and_return_conditional_losses_5920STU6789:VWXYZ[Aข>
7ข4
*'
conv1d_input?????????
p 

 
ช "2ข/
(%
0??????????????????
 ะ
D__inference_sequential_layer_call_and_return_conditional_losses_6549STU6789:VWXYZ[;ข8
1ข.
$!
inputs?????????
p

 
ช "2ข/
(%
0??????????????????
 ะ
D__inference_sequential_layer_call_and_return_conditional_losses_6910STU6789:VWXYZ[;ข8
1ข.
$!
inputs?????????
p 

 
ช "2ข/
(%
0??????????????????
 ฎ
)__inference_sequential_layer_call_fn_6027STU6789:VWXYZ[Aข>
7ข4
*'
conv1d_input?????????
p

 
ช "%"??????????????????ฎ
)__inference_sequential_layer_call_fn_6133STU6789:VWXYZ[Aข>
7ข4
*'
conv1d_input?????????
p 

 
ช "%"??????????????????ง
)__inference_sequential_layer_call_fn_6955zSTU6789:VWXYZ[;ข8
1ข.
$!
inputs?????????
p

 
ช "%"??????????????????ง
)__inference_sequential_layer_call_fn_7000zSTU6789:VWXYZ[;ข8
1ข.
$!
inputs?????????
p 

 
ช "%"???????????????????
"__inference_signature_wrapper_6188ทSTU6789:VWXYZ[IขF
ข 
?ช<
:
conv1d_input*'
conv1d_input?????????"TชQ
O
time_distributed_296
time_distributed_2??????????????????ั
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_8070XYEขB
;ข8
.+
inputs??????????????????
p

 
ช "3ข0
)&
0??????????????????
 ั
L__inference_time_distributed_1_layer_call_and_return_conditional_losses_8092XYEขB
;ข8
.+
inputs??????????????????
p 

 
ช "3ข0
)&
0??????????????????
 จ
1__inference_time_distributed_1_layer_call_fn_8101sXYEขB
;ข8
.+
inputs??????????????????
p

 
ช "&#??????????????????จ
1__inference_time_distributed_1_layer_call_fn_8110sXYEขB
;ข8
.+
inputs??????????????????
p 

 
ช "&#??????????????????ฯ
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_8131Z[EขB
;ข8
.+
inputs??????????????????
p

 
ช "2ข/
(%
0??????????????????
 ฯ
L__inference_time_distributed_2_layer_call_and_return_conditional_losses_8152Z[EขB
;ข8
.+
inputs??????????????????
p 

 
ช "2ข/
(%
0??????????????????
 ง
1__inference_time_distributed_2_layer_call_fn_8161rZ[EขB
;ข8
.+
inputs??????????????????
p

 
ช "%"??????????????????ง
1__inference_time_distributed_2_layer_call_fn_8170rZ[EขB
;ข8
.+
inputs??????????????????
p 

 
ช "%"??????????????????อ
J__inference_time_distributed_layer_call_and_return_conditional_losses_8008VWDขA
:ข7
-*
inputs??????????????????

p

 
ช "3ข0
)&
0??????????????????
 อ
J__inference_time_distributed_layer_call_and_return_conditional_losses_8030VWDขA
:ข7
-*
inputs??????????????????

p 

 
ช "3ข0
)&
0??????????????????
 ฅ
/__inference_time_distributed_layer_call_fn_8039rVWDขA
:ข7
-*
inputs??????????????????

p

 
ช "&#??????????????????ฅ
/__inference_time_distributed_layer_call_fn_8048rVWDขA
:ข7
-*
inputs??????????????????

p 

 
ช "&#??????????????????