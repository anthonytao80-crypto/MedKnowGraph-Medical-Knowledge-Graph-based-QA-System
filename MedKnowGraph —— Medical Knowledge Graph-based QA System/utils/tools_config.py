from langchain_core.tools import tool
from py2neo import Graph

# --------------------  核心工具函数 --------------------

def get_tools():
    """构建 LangChain 工具列表"""

    class AnswerSearcher:
        def __init__(self):
            self.g = Graph("bolt://localhost:7687", auth=("neo4j", "Aa123456"))
            self.num_limit = 20

        def search_main(self, sqls):
            """执行 Cypher 查询"""
            retrive = []
            for sql_ in sqls:
                queries = sql_['sql']
                for query in queries:
                    ress = self.g.run(query).data()
                    retrive.extend(ress)
            return retrive


    class QuestionPaser:
        """生成 Cypher 查询"""

        def build_entitydict(self, args):
            entity_dict = {}
            for arg, types in args.items():
                for type in types:
                    if type not in entity_dict:
                        entity_dict[type] = [arg]
                    else:
                        entity_dict[type].append(arg)
            return entity_dict

        def parser_main(self, res_classify):
            args = res_classify['args']
            entity_dict = self.build_entitydict(args)
            question_types = res_classify['question_types']
            sqls = []
            for question_type in question_types:
                sql_ = {'question_type': question_type}
                sql_['sql'] = self.sql_transfer(question_type, entity_dict.get('disease'))
                sqls.append(sql_)
            return sqls

        def sql_transfer(self, question_type, entities):
            if not entities:
                return []
            sql = []
            # 查询疾病的原因
            if question_type == 'disease_cause':
                sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cause".format(i) for i in entities]

            # 查询疾病的防御措施
            elif question_type == 'disease_prevent':
                sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.prevent".format(i) for i in entities]

            # 查询疾病的持续时间
            elif question_type == 'disease_lasttime':
                sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cure_lasttime".format(i) for i in
                       entities]

            # 查询疾病的治愈概率
            elif question_type == 'disease_cureprob':
                sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cured_prob".format(i) for i in entities]

            # 查询疾病的治疗方式
            elif question_type == 'disease_cureway':
                sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cure_way".format(i) for i in entities]

            # 查询疾病的易发人群
            elif question_type == 'disease_easyget':
                sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.easy_get".format(i) for i in entities]

            # 查询疾病的相关介绍
            elif question_type == 'disease_desc':
                sql = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.desc".format(i) for i in entities]

            # 查询疾病有哪些症状
            elif question_type == 'disease_symptom':
                sql = [
                    "MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]

            # 查询症状会导致哪些疾病
            elif question_type == 'symptom_disease':
                sql = [
                    "MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where n.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]

            # 查询疾病的并发症
            elif question_type == 'disease_acompany':
                sql1 = [
                    "MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where m.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql2 = [
                    "MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where n.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql = sql1 + sql2
            # 查询疾病的忌口
            elif question_type == 'disease_not_food':
                sql = [
                    "MATCH (m:Disease)-[r:no_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]

            # 查询疾病建议吃的东西
            elif question_type == 'disease_do_food':
                sql1 = [
                    "MATCH (m:Disease)-[r:do_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql2 = [
                    "MATCH (m:Disease)-[r:recommand_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql = sql1 + sql2

            # 已知忌口查疾病
            elif question_type == 'food_not_disease':
                sql = [
                    "MATCH (m:Disease)-[r:no_eat]->(n:Food) where n.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]

            # 已知推荐查疾病
            elif question_type == 'food_do_disease':
                sql1 = [
                    "MATCH (m:Disease)-[r:do_eat]->(n:Food) where n.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql2 = [
                    "MATCH (m:Disease)-[r:recommand_eat]->(n:Food) where n.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql = sql1 + sql2

            # 查询疾病常用药品－药品别名记得扩充
            elif question_type == 'disease_drug':
                sql1 = [
                    "MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql2 = [
                    "MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql = sql1 + sql2

            # 已知药品查询能够治疗的疾病
            elif question_type == 'drug_disease':
                sql1 = [
                    "MATCH (m:Disease)-[r:common_drug]->(n:Drug) where n.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql2 = [
                    "MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where n.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]
                sql = sql1 + sql2
            # 查询疾病应该进行的检查
            elif question_type == 'disease_check':
                sql = [
                    "MATCH (m:Disease)-[r:need_check]->(n:Check) where m.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]

            # 已知检查查询疾病
            elif question_type == 'check_disease':
                sql = [
                    "MATCH (m:Disease)-[r:need_check]->(n:Check) where n.name = '{0}' return m.name, r.name, n.name".format(
                        i) for i in entities]

            return sql
    # --------------------  LangChain 工具 --------------------
    @tool
    def retriever_tool(entities: list, classify: str):
        """
        基于疾病知识图谱的查询工具
        参数:
            entities: 命名实体列表
            classify: 问题类型，从以下类别中选取：
            从以下类别中选取：
                'disease_cause': 查询疾病的原因
                'disease_prevent': 查询疾病的防御措施
                'disease_lasttime': 查询疾病的持续时间
                'disease_cureprob': 查询疾病的治愈概率
                'disease_cureway': 查询疾病的治疗方式
                'disease_easyget': 查询疾病的易发人群
                'disease_desc': 查询疾病的相关介绍
                'disease_symptom': 查询疾病有哪些症状
                'symptom_disease': 查询症状会导致哪些疾病
                'disease_acompany': 查询疾病的并发症
                'disease_not_food': 查询疾病的忌口
                'disease_do_food': 查询疾病建议吃的东西
                'food_not_disease': 已知忌口查疾病
                'food_do_disease': 已知推荐食物查疾病
                'disease_drug': 查询疾病常用药品
                'drug_disease': 已知药品查询能够治疗的疾病
                'disease_check': 查询疾病应该进行的检查
                'check_disease': 已知检查查询疾病
        返回:
            查询结果 (list[dict])
        """
        parser = QuestionPaser()
        searcher = AnswerSearcher()

        # 1. 构造输入
        classify_input = {
            "args": {entity: ["disease"] for entity in entities},  # 假设实体类型都是疾病
            "question_types": [classify]
        }

        # 2. 生成查询语句
        sqls = parser.parser_main(classify_input)

        # 3. 执行查询
        results = searcher.search_main(sqls)

        return results

    return [retriever_tool]

# test
# tools = get_tools()
# retriever_tool = tools[0]
# res = retriever_tool.run({
#     "entities": ['肺气肿'],
#     "classify": 'disease_not_food'
# })
#
# print(res)