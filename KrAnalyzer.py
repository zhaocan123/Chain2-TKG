from collections import defaultdict
import math

class KrAnalyzer:
    def __init__(self, triples, metrics_set, theta=0.3):
        """
        :param triples: 列表，每个元素为 (s, r, o)，s和o为实体名称字符串
        :param metrics_set: 集合，包含所有韧性评估指标实体名称
        :param theta: 语义相似度阈值，默认0.3
        """
        self.triples = triples
        self.metrics_set = metrics_set
        self.theta = theta
        self.entities = set()
        for s, _, o in triples:
            self.entities.add(s)
            self.entities.add(o)
        self.topology = defaultdict(set)          # 正向邻接表
        self.reverse_topology = defaultdict(set)  # 反向邻接表
        self.build_topology()

        # 并查集计算分支标签
        self.parent = {e: e for e in self.entities}
        self.rank = {e: 0 for e in self.entities}
        self.build_branch_label()

    def build_topology(self):
        """构建正向和反向邻接表（有向图）"""
        for s, _, o in self.triples:
            self.topology[s].add(o)
            self.reverse_topology[o].add(s)

    def find(self, x):
        """并查集查找"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """并查集合并"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

    def build_branch_label(self):
        """基于无向连通分量计算分支标签"""
        # 将所有有向边视为无向边进行合并
        for s, _, o in self.triples:
            self.union(s, o)
        # 每个实体的分支标签即其根节点
        self.branch_label = {e: self.find(e) for e in self.entities}

    def shortest_path_length(self, s, o):
        """
        计算有向图中从s到o的最短路径跳数（仅考虑有向边）
        只关心0,1,2，更远返回inf
        """
        if s == o:
            return 0
        if o in self.topology[s]:
            return 1
        # 检查是否存在中间节点t，使得s->t且t->o
        for t in self.topology[s]:
            if o in self.topology[t]:
                return 2
        return math.inf

    def semantic_similarity(self, s, o):
        """基于下划线分词计算Jaccard相似度"""
        set_s = set(s.lower().split('_'))
        set_o = set(o.lower().split('_'))
        if not set_s or not set_o:
            return 0.0
        inter = len(set_s & set_o)
        union = len(set_s | set_o)
        return inter / union if union else 0.0

    def analyze(self):
        """
        执行关系分类，返回新的三元组列表 [(s, relation, o)]
        其中relation为 'hierarchy', 'coupling', 'minimal' 之一
        """
        result_dict = {}  # key: (s, o), value: relation (按优先级覆盖)

        for s, _, o in self.triples:
            # 先根据距离判断
            dist = self.shortest_path_length(s, o)
            kr = None

            if dist == 1:
                kr = 'hierarchy'
            elif dist == 2:
                # 检查共同父节点
                common_parents = self.reverse_topology[s] & self.reverse_topology[o]
                commonP = bool(common_parents)
                # 检查是否跨分支
                crossB = (self.branch_label.get(s) != self.branch_label.get(o))
                if (commonP or crossB) and self.semantic_similarity(s, o) < self.theta:
                    kr = 'coupling'

            # 若尾实体是韧性指标，直接标记为 minimal（优先级最高）
            if o in self.metrics_set:
                kr = 'minimal'

            # 仅当有有效kr时才记录
            if kr is not None:
                # 如果已存在该实体对，比较优先级：minimal > coupling > hierarchy
                if (s, o) in result_dict:
                    old = result_dict[(s, o)]
                    # 优先级权重：minimal=3, coupling=2, hierarchy=1
                    priority = {'minimal': 3, 'coupling': 2, 'hierarchy': 1}
                    if priority[kr] > priority[old]:
                        result_dict[(s, o)] = kr
                else:
                    result_dict[(s, o)] = kr

        # 转换为列表
        new_triples = [(s, kr, o) for (s, o), kr in result_dict.items()]
        return new_triples


# ================== 使用示例 ==================
if __name__ == '__main__':
    # 模拟输入：你之前生成的知识单元（三元组）
    # 假设已有 triples 列表，每个元素 (head, relation, tail)
    # metrics_set 即你定义的 metrics 列表的集合

    # 这里用一些简单示例演示
    sample_triples = [
        ("steel_price_fluctuation", "some_rel", "delivery_delay_days"),
        ("steel_price_fluctuation", "some_rel", "MES_manufacturing_execution_system"),
        ("MES_manufacturing_execution_system", "some_rel", "delivery_delay_days"),
        ("shipyard_management", "some_rel", "MES_manufacturing_execution_system"),
        ("shipyard_management", "some_rel", "steel_supplier"),  # 无直接关系，但可能有2-hop
    ]

    metrics_set = {
        "delivery_delay_days", "extra_cost", "quality_pass_rate", "supply_chain_visibility_rate",
        "blockchain_query_response_time", "production_volatility", "customer_satisfaction",
        "emergency_response_speed", "redundancy_capacity", "network_resilience_score",
        "risk_alert_accuracy", "system_availability", "information_transparency",
        "collaboration_efficiency", "recovery_time"
    }

    analyzer = KrAnalyzer(sample_triples, metrics_set, theta=0.3)
    classified = analyzer.analyze()

    for s, kr, o in classified:
        print(f"{s} --{kr}--> {o}")