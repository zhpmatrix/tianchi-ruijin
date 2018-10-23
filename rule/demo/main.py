from pyltp import Segmentor
segmentor = Segmentor()
#segmentor.load_with_lexicon('../model/cws.model', 'dict.txt')
segmentor.load('../model/cws.model')
print('\t'.join(segmentor.segment('这是一种化学物质')))
