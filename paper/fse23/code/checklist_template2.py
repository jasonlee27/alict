air_noun = [(*@\label{code:tempex:airnoun:1}@*)
    'flight', 'seat', 'pilot', 'staff', ... 
](*@\label{code:tempex:airnoun:2}@*)
pos_adj = [(*@\label{code:tempex:posadj:1}@*)
    'good', 'great', 'excellent', 'amazing', ...
](*@\label{code:tempex:posadj:2}@*)
neg_adj = [(*@\label{code:tempex:negadj:1}@*)
    'awful', 'bad', 'horrible', 'weird', ...
](*@\label{code:tempex:negadj:2}@*)
t = editor.template('{it} {air_noun} {be} {pos_adj}.', (*@\label{code:tempex:template:1}@*)
                    it=['The', 'This', 'That'], be=['is', 'was'], labels=2, save=True)(*@\label{code:tempex:itbe}@*)
t += editor.template('{it} {be} {a:pos_adj} {air_noun}.', 
                     it=['It', 'This', 'That'], be=['is', 'was'], labels=2, save=True)
t += ...
t += editor.template('{it} {be} {a:neg_adj} {air_noun}.', 
                     it=['It', 'This', 'That'], be=['is', 'was'], labels=0, save=True)(*@\label{code:tempex:template:2}@*)

