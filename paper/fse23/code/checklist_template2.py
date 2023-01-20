pos_adj = [(*@\label{code:tempex:posadj:1}@*)
    'good', 'great', 'excellent', 'amazing', ...
](*@\label{code:tempex:posadj:2}@*)
neg_adj = [(*@\label{code:tempex:negadj:1}@*)
    'awful', 'bad', 'horrible', 'weird', ...
](*@\label{code:tempex:negadj:2}@*)
change = ['but', 'even though', 'although'](*@\label{code:tempex:change}@*)
t = editor.template([(*@\label{code:tempex:template:1}@*)
    'I used to think this airline was {neg_adj}, {change} now I think it is {pos_adj}.',
    'I think this airline is {pos_adj}, {change} I used to think it was {neg_adj}.',
    'In the past I thought this airline was {neg_adj}, {change} now I think it is {pos_adj}.',
    'I think this airline is {pos_adj}, {change} in the past I thought it was {neg_adj}.',
] ,
change=change, ... ,labels=2)(*@\label{code:tempex:template:2}@*)
