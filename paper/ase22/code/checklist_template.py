air_noun = [(*@\label{code:tempex:airnoun:1}@*)
    'flight', 'seat', 'pilot', 'staff', 
    'service', 'customer service', 'aircraft', 'plane', 
    'food', 'cabin crew', 'company', 'airline', 'crew'
](*@\label{code:tempex:airnoun:2}@*)
pos_adj = [(*@\label{code:tempex:posadj:1}@*)
    'good', 'great', 'excellent', 'amazing', 
    'extraordinary', 'beautiful', 'fantastic', 'nice', 
    'incredible', 'exceptional', 'awesome', 'perfect', 
    'fun', 'happy', 'adorable', 'brilliant', 'exciting', 
    'sweet', 'wonderful'
](*@\label{code:tempex:posadj:2}@*)
neg_adj = [(*@\label{code:tempex:negadj:1}@*)
    'awful', 'bad', 'horrible', 'weird', 
    'rough', 'lousy', 'unhappy', 'average', 
    'difficult', 'poor', 'sad', 'frustrating', 
    'hard', 'lame', 'nasty', 'annoying', 'boring', 
    'creepy', 'dreadful', 'ridiculous', 'terrible', 
    'ugly', 'unpleasant'
](*@\label{code:tempex:negadj:2}@*)

t = editor.template('{it} {air_noun} {be} {pos_adj}.', (*@\label{code:tempex:template:1}@*)
                    it=['The', 'This', 'That'], be=['is', 'was'], labels=2, save=True)(*@\label{code:tempex:itbe}@*)
t += editor.template('{it} {be} {a:pos_adj} {air_noun}.', 
                     it=['It', 'This', 'That'], be=['is', 'was'], labels=2, save=True)
t += editor.template('{i} {pos_verb} {the} {air_noun}.', 
                     i=['I', 'We'], the=['this', 'that', 'the'], labels=2, save=True)
t += editor.template('{it} {air_noun} {be} {neg_adj}.', 
                     it=['That', 'This', 'The'], be=['is', 'was'], labels=0, save=True)
t += editor.template('{it} {be} {a:neg_adj} {air_noun}.', 
                     it=['It', 'This', 'That'], be=['is', 'was'], labels=0, save=True)
t += editor.template('{i} {neg_verb} {the} {air_noun}.', 
                     i=['I', 'We'], the=['this', 'that', 'the'], labels=0, save=True)(*@\label{code:tempex:template:2}@*)
