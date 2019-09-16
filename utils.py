import wikipediaapi as wa
import textacy
from textacy import preprocessing as pp
import spacy

def load_nlp():
    disable = ['tagger', 'parser', 'ner']
    nlp = spacy.load("en_core_web_md", disable=disable)
    return nlp

def display_record_info(cats, corpus):
    for cat in cats:
        print(cat.upper())
        subset = corpus.get_by_category(cat)
        lens = [*map(len, subset)]
        print("Num docs = {}, Avg words per doc = {}".format(len(lens), sum(lens) / len(lens)))
        print()
    print("Total Docs = {}, Avg words per doc = {}".format(len(corpus),sum((len(d) for d in corpus))/len(corpus)))

class WikiApiWrapper(wa.Wikipedia):
    def __init__(self):
        super().__init__(language='en', extract_format=wa.ExtractFormat.WIKI)

    def get_category_records(self, cat, limit=None):
        cat_page = self.page(cat)
        children = list(set(cat_page.categorymembers.values()))
        if limit:
            children = children[:limit]
        for c in children:
            res = self.process_categorymember(c, cat)
            if res: 
                yield res
    
    def process_categorymember(self, page, cat, min_len=300):
        if page.ns == 0 and page.language == 'en' and page.exists:
            txt = page.summary
            
            if len(txt) < min_len:
                return None
            metadata = {
                "category": cat,
                "title": page.title,
                "id": page.pageid
            }
            return txt, metadata
    
    
class CorpusWrapper(textacy.Corpus):
    LIMIT = None

    def __init__(self, nlp):
        self._nlp = nlp
        super().__init__(nlp)
    
    def get_custom(self, doc, key):
        return doc._.meta[key]
    
    def title(self, doc):
        return self.get_custom(doc, 'title')
    
    def category(self, doc):
        return self.get_custom(doc, 'category')

    def get_doc(self, id):
        match_func = lambda _doc: self.get_custom(_doc, 'id') == id
        return self.get(match_func)
    
    def get_by_category(self, cat):
        match_func = lambda _doc: self.get_custom(_doc, 'category') == cat
        return self.get(match_func)


    def process_records(self, records, min_words=1, max_words=2**32):
        for txt, metadata in records:
            txt = self.preprocess_text(txt)
            self.add_record((txt, metadata))
            record = self[-1]
            if not min_words <= len(record) <= max_words:
                del self[-1]
    
    def preprocess_text(self, txt):
        txt = pp.normalize.normalize_whitespace(txt)
        doc_tokens = [tok.text.lower() for tok in self._nlp(txt) if \
                      not tok.is_stop 
                      and 
                      not tok.is_punct
                     ]
        txt = ' '.join(doc_tokens)
        return txt
    
    def load_records(self, categories, wa):
        for cat in categories:
            print("Loading articles under %s"%cat)
            records = list(wa.get_category_records(cat, limit=self.LIMIT))
            self.process_records(records, min_words=35, max_words=120)
        print()
