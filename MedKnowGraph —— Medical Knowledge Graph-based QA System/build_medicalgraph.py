# Selected and modified from open-source project (Liu Huanyong, Institute of Software, Chinese Academy of Sciences)
# Dataset obtained from web-crawled medical data
import os
import json
from py2neo import Graph, Node

class MedicalGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'data/medical2.json')
        self.g = Graph("bolt://localhost:7687", auth=("neo4j", "Aa123456"))

    '''Read data from file'''
    def read_nodes(self):
        # There are 7 types of entities
        drugs = []          # medical drugs
        foods = []          # foods
        checks = []         # medical checks
        departments = []    # medical departments
        producers = []      # drug producers
        diseases = []       # disease names
        symptoms = []       # symptoms

        disease_infos = []  # disease information dictionary

        # Define relationship containers
        rels_department = []     # department-to-department relationships
        rels_noteat = []         # disease - forbidden foods
        rels_doeat = []          # disease - suitable foods
        rels_recommandeat = []   # disease - recommended foods
        rels_commonddrug = []    # disease - common drugs
        rels_recommanddrug = []  # disease - recommended drugs
        rels_check = []          # disease - diagnostic checks
        rels_drug_producer = []  # producer - drug relationships

        rels_symptom = []        # disease - symptom relationships
        rels_acompany = []       # disease - comorbidity relationships
        rels_category = []       # disease - department relationships

        count = 0
        for data in open(self.data_path):
            disease_dict = {}
            count += 1
            print(count)
            data_json = json.loads(data)
            disease = data_json['name']
            disease_dict['name'] = disease
            diseases.append(disease)
            disease_dict['desc'] = ''
            disease_dict['prevent'] = ''
            disease_dict['cause'] = ''
            disease_dict['easy_get'] = ''
            disease_dict['cure_department'] = ''
            disease_dict['cure_way'] = ''
            disease_dict['cure_lasttime'] = ''
            disease_dict['symptom'] = ''
            disease_dict['cured_prob'] = ''

            if 'symptom' in data_json:
                symptoms += data_json['symptom']
                for symptom in data_json['symptom']:
                    rels_symptom.append([disease, symptom])

            if 'acompany' in data_json:
                for acompany in data_json['acompany']:
                    rels_acompany.append([disease, acompany])

            if 'desc' in data_json:
                disease_dict['desc'] = data_json['desc']

            if 'prevent' in data_json:
                disease_dict['prevent'] = data_json['prevent']

            if 'cause' in data_json:
                disease_dict['cause'] = data_json['cause']

            if 'get_prob' in data_json:
                disease_dict['get_prob'] = data_json['get_prob']

            if 'easy_get' in data_json:
                disease_dict['easy_get'] = data_json['easy_get']

            if 'cure_department' in data_json:
                cure_department = data_json['cure_department']
                if len(cure_department) == 1:
                    rels_category.append([disease, cure_department[0]])
                if len(cure_department) == 2:
                    big = cure_department[0]
                    small = cure_department[1]
                    rels_department.append([small, big])
                    rels_category.append([disease, small])

                disease_dict['cure_department'] = cure_department
                departments += cure_department

            if 'cure_way' in data_json:
                disease_dict['cure_way'] = data_json['cure_way']

            if 'cure_lasttime' in data_json:
                disease_dict['cure_lasttime'] = data_json['cure_lasttime']

            if 'cured_prob' in data_json:
                disease_dict['cured_prob'] = data_json['cured_prob']

            if 'common_drug' in data_json:
                common_drug = data_json['common_drug']
                for drug in common_drug:
                    rels_commonddrug.append([disease, drug])
                drugs += common_drug

            if 'recommand_drug' in data_json:
                recommand_drug = data_json['recommand_drug']
                drugs += recommand_drug
                for drug in recommand_drug:
                    rels_recommanddrug.append([disease, drug])

            if 'not_eat' in data_json:
                not_eat = data_json['not_eat']
                for _not in not_eat:
                    rels_noteat.append([disease, _not])
                foods += not_eat

                do_eat = data_json['do_eat']
                for _do in do_eat:
                    rels_doeat.append([disease, _do])
                foods += do_eat

                recommand_eat = data_json['recommand_eat']
                for _recommand in recommand_eat:
                    rels_recommandeat.append([disease, _recommand])
                foods += recommand_eat

            if 'check' in data_json:
                check = data_json['check']
                for _check in check:
                    rels_check.append([disease, _check])
                checks += check

            if 'drug_detail' in data_json:
                drug_detail = data_json['drug_detail']
                producer = [i.split('(')[0] for i in drug_detail]
                rels_drug_producer += [[i.split('(')[0], i.split('(')[-1].replace(')', '')] for i in drug_detail]
                producers += producer

            disease_infos.append(disease_dict)

        return (
            set(drugs), set(foods), set(checks), set(departments), set(producers),
            set(symptoms), set(diseases), disease_infos,
            rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department,
            rels_commonddrug, rels_drug_producer, rels_recommanddrug,
            rels_symptom, rels_acompany, rels_category
        )

    '''Create entity nodes'''
    def create_node(self, label, nodes):
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.g.create(node)
            count += 1
            print(count, len(nodes))
        return

    '''Create disease nodes (the central nodes in the knowledge graph)'''
    def create_diseases_nodes(self, disease_infos):
        count = 0
        for disease_dict in disease_infos:
            node = Node(
                "Disease",
                name=disease_dict['name'],
                desc=disease_dict['desc'],
                prevent=disease_dict['prevent'],
                cause=disease_dict['cause'],
                easy_get=disease_dict['easy_get'],
                cure_lasttime=disease_dict['cure_lasttime'],
                cure_department=disease_dict['cure_department'],
                cure_way=disease_dict['cure_way'],
                cured_prob=disease_dict['cured_prob']
            )
            self.g.create(node)
            count += 1
            print(count)
        return

    '''Create entity node schemas for the medical knowledge graph'''
    def create_graphnodes(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, \
        rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, \
        rels_commonddrug, rels_drug_producer, rels_recommanddrug, rels_symptom, \
        rels_acompany, rels_category = self.read_nodes()

        self.create_diseases_nodes(disease_infos)
        self.create_node('Drug', Drugs)
        print(len(Drugs))
        self.create_node('Food', Foods)
        print(len(Foods))
        self.create_node('Check', Checks)
        print(len(Checks))
        self.create_node('Department', Departments)
        print(len(Departments))
        self.create_node('Producer', Producers)
        print(len(Producers))
        self.create_node('Symptom', Symptoms)
        return

    '''Create all relationships between entities'''
    def create_graphrels(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, \
        rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, \
        rels_commonddrug, rels_drug_producer, rels_recommanddrug, rels_symptom, \
        rels_acompany, rels_category = self.read_nodes()

        self.create_relationship('Disease', 'Food', rels_recommandeat, 'recommand_eat', 'Recommended Food')
        self.create_relationship('Disease', 'Food', rels_noteat, 'no_eat', 'Forbidden Food')
        self.create_relationship('Disease', 'Food', rels_doeat, 'do_eat', 'Suitable Food')
        self.create_relationship('Department', 'Department', rels_department, 'belongs_to', 'Belongs To')
        self.create_relationship('Disease', 'Drug', rels_commonddrug, 'common_drug', 'Common Drug')
        self.create_relationship('Producer', 'Drug', rels_drug_producer, 'drugs_of', 'Produces Drug')
        self.create_relationship('Disease', 'Drug', rels_recommanddrug, 'recommand_drug', 'Recommended Drug')
        self.create_relationship('Disease', 'Check', rels_check, 'need_check', 'Required Check')
        self.create_relationship('Disease', 'Symptom', rels_symptom, 'has_symptom', 'Has Symptom')
        self.create_relationship('Disease', 'Disease', rels_acompany, 'acompany_with', 'Comorbidity')
        self.create_relationship('Disease', 'Department', rels_category, 'belongs_to', 'Belongs To')

    '''Create relationships between entity nodes'''
    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        # Remove duplicates
        set_edges = ['###'.join(edge) for edge in edges]
        total = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "MATCH(p:%s),(q:%s) WHERE p.name='%s' AND q.name='%s' CREATE (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, total)
            except Exception as e:
                print(e)
        return

    '''Export extracted data to text files'''
    def export_data(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, \
        rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, \
        rels_commonddrug, rels_drug_producer, rels_recommanddrug, rels_symptom, \
        rels_acompany, rels_category = self.read_nodes()

        f_drug = open('drug.txt', 'w+')
        f_food = open('food.txt', 'w+')
        f_check = open('check.txt', 'w+')
        f_department = open('department.txt', 'w+')
        f_producer = open('producer.txt', 'w+')
        f_symptom = open('symptoms.txt', 'w+')
        f_disease = open('disease.txt', 'w+')

        f_drug.write('\n'.join(list(Drugs)))
        f_food.write('\n'.join(list(Foods)))
        f_check.write('\n'.join(list(Checks)))
        f_department.write('\n'.join(list(Departments)))
        f_producer.write('\n'.join(list(Producers)))
        f_symptom.write('\n'.join(list(Symptoms)))
        f_disease.write('\n'.join(list(Diseases)))

        f_drug.close()
        f_food.close()
        f_check.close()
        f_department.close()
        f_producer.close()
        f_symptom.close()
        f_disease.close()
        return


if __name__ == '__main__':
    handler = MedicalGraph()
    # handler.export_data()
    handler.create_graphnodes()
    handler.create_graphrels()

