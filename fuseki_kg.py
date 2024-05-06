from SPARQLWrapper import SPARQLWrapper, POST, DIGEST, JSON
import re

class Fuseki():
    def __init__(self, endpoint_url: str, username: str, password: str):
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setHTTPAuth(DIGEST)
        self.sparql.setCredentials(user=username, passwd=password)
        self.sparql.setMethod(POST)
        self.sparql.setMethod(JSON)
        self.prefix_list = ["http://dasi/eid/", "http://dasi/relation/", "http://dasi/property/", "http://dasi/pid/"] # examples
        self.prefix = """PREFIX eid:<http://dasi/eid/>\nPREFIX relation:<http://dasi/relation/>\nPREFIX property:<http://dasi/property/>\nPREFIX pid:<http://dasi/pid/>"""
        self.triplet_list = []

    def execute_query(self, query: str):
        self.sparql.setQuery(query)
        results = self.sparql.query().convert()
        return results
    
    def extract_triple_from_query(self, head, query_results):
        triple_list = [[head, i['relation']['value'], i['tail']['value']] for i in query_results]
        pattern = r"^(?:" + "|".join(self.prefix_list) + r").*"
        relation_node_list = list(set([item[2] for item in triple_list if bool(re.match(pattern, item[2]))]))
        return {'triple': triple_list, 'related_node': relation_node_list}
    
    def query_by_name(self, name: str):
        query = self.prefix + """select ?relation ?x where {{?e property:name "{0}". ?e ?relation ?x}}""".format(name)
        query_results = self.execute_query(query)
        if query_results['results']['bindings'] == []:
            return None
        info_dict = self.extract_triple_from_query(name, query_results)
        return info_dict
    
    def query_by_id(self, id: str):
        query = self.prefix + """select ?relation ?x where {{<{0}> ?relation ?x.}}""".format(id)
        query_results = self.execute_query(query)
        if query_results['results']['bindings'] == []:
            return None
        else:
            try:
                name = query_results['results']['bindings'][0]['name']['value']
            except IndexError:
                return "untitled"
        info_dict = self.extract_triple_from_query(name, query_results)
        return info_dict
    
    def expand_relation(self, reified_relation_id):
        query = self.prefix + """select ?from_relation ?from_node ?to_relation ?to_node where {{<{0}> ?to_relation ?to_node. ?from_node ?from_relation <{0}>.}}""".format(reified_relation_id)
        query_results = self.execute_query(query)['results']['bindings'][0]
        from_node_id = query_results['from_node']['value']
        to_node_id = query_results['to_node']['value']
        from_name = self.query_by_id(from_node_id)['triple'][0][0]
        to_name = self.query_by_id(to_node_id)['triple'][0][0]
        triple_list = [[from_name, relation, to_name] for relation in query_results['from_relation']+query_results['to_relation']]
        return triple_list
    
    def update_query(self, update_query):
        self.test_iq = update_query
        self.sparql.setQuery(update_query)
        self.sparql.query()