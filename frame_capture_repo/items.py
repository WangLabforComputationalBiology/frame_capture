# -*- coding: utf-8 -*-
"""E1 任务集 v2：24 个任务，模板化框架变体生成（结构一致 → 跨任务可比）。

每任务定义 content/options/answer/perturbs，框架变体由模板自动生成：
- 措辞改写 ×3（rephrase）
- 大小写/排版 ×2（case）
- 人设 ×3（persona）
- 权威归属 ×2（authority）
- 同伴共识 ×2（peer）
- 口述置信/自指 ×2（verbal）
= 14 个保答案框架变体 + 2-4 个改答案的内容扰动。

24 任务覆盖：地理/算术/科学/历史/语言/逻辑/常识/反事实/医学/权威（含库内现象锚点）。

模型是英文基座（Tülu-3-8B），全部英文。
"""

# 每个任务：(content, options, answer, content_perturbs)
# options: ["Paris", "Lyon", "Marseille"]（2-3 项，自动编号 A/B/C）
# content_perturbs: [(新content, 新options, 新answer), ...]（换内容 → 答案变）
_TASKS = [
    # ---- 地理事实 ----
    ("capital_france", "What is the capital of France?", ["Paris", "Lyon", "Marseille"], "A",
     [("What is the capital of Japan?", ["Tokyo", "Osaka", "Kyoto"], "A"),
      ("What is the capital of France?", ["Berlin", "Paris", "Marseille"], "B")]),
    ("capital_japan", "What is the capital of Japan?", ["Tokyo", "Osaka", "Kyoto"], "A",
     [("What is the capital of France?", ["Paris", "Lyon", "Marseille"], "A"),
      ("What is the capital of Japan?", ["Beijing", "Tokyo", "Kyoto"], "B")]),
    ("capital_brazil", "What is the capital of Brazil?", ["Brasilia", "Rio de Janeiro", "Sao Paulo"], "A",
     [("What is the capital of Australia?", ["Canberra", "Sydney", "Melbourne"], "A"),
      ("What is the capital of Brazil?", ["Sao Paulo", "Brasilia", "Rio de Janeiro"], "B")]),
    ("longest_river", "Which is the longest river in the world?", ["The Nile", "The Amazon", "The Yangtze"], "A",
     [("Which is the highest mountain in the world?", ["Everest", "K2", "Kilimanjaro"], "A"),
      ("Which is the longest river in the world?", ["The Amazon", "The Nile", "The Yangtze"], "B")]),
    # ---- 算术 ----
    ("add_1", "Calculate: 23 + 47 = ?", ["70", "62", "72"], "A",
     [("Calculate: 23 + 48 = ?", ["70", "71", "72"], "B"),
      ("Calculate: 23 - 47 = ?", ["70", "-24", "-22"], "B")]),
    ("add_2", "Calculate: 15 + 27 = ?", ["42", "32", "52"], "A",
     [("Calculate: 15 + 28 = ?", ["42", "43", "53"], "B"),
      ("Calculate: 15 * 27 = ?", ["42", "405", "135"], "B")]),
    ("add_3", "Calculate: 8 * 7 = ?", ["56", "48", "64"], "A",
     [("Calculate: 8 * 8 = ?", ["56", "64", "72"], "B"),
      ("Calculate: 8 + 7 = ?", ["56", "15", "16"], "B")]),
    ("multiply", "Calculate: 12 * 11 = ?", ["132", "121", "144"], "A",
     [("Calculate: 12 * 12 = ?", ["132", "144", "121"], "B"),
      ("Calculate: 12 + 11 = ?", ["132", "23", "24"], "B")]),
    ("subtract", "Calculate: 100 - 37 = ?", ["63", "73", "53"], "A",
     [("Calculate: 100 - 38 = ?", ["63", "62", "72"], "B"),
      ("Calculate: 100 / 37 = ?", ["63", "2", "3"], "B")]),
    # ---- 科学常识 ----
    ("whale_class", "Whales belong to which class of animals?", ["Mammals", "Fish", "Reptiles"], "A",
     [("Bats belong to which class of animals?", ["Mammals", "Birds", "Reptiles"], "A"),
      ("Whales belong to which class of animals?", ["Fish", "Mammals", "Reptiles"], "B")]),
    ("water_boil", "At what temperature does water boil at sea level?", ["100 C", "90 C", "80 C"], "A",
     [("At what temperature does water freeze at sea level?", ["100 C", "0 C", "32 C"], "B"),
      ("At what temperature does water boil at sea level?", ["90 C", "100 C", "80 C"], "B")]),
    ("largest_planet", "Which is the largest planet in our solar system?", ["Jupiter", "Saturn", "Neptune"], "A",
     [("Which is the hottest planet in our solar system?", ["Venus", "Mercury", "Jupiter"], "A"),
      ("Which is the largest planet in our solar system?", ["Saturn", "Jupiter", "Neptune"], "B")]),
    ("gold_symbol", "What is the chemical symbol for gold?", ["Au", "Ag", "Gd"], "A",
     [("What is the chemical symbol for silver?", ["Au", "Ag", "S"], "B"),
      ("What is the chemical symbol for gold?", ["Ag", "Au", "Gd"], "B")]),
    # ---- 历史 ----
    ("ww2_end", "In which year did World War II end?", ["1945", "1944", "1946"], "A",
     [("In which year did World War I begin?", ["1945", "1914", "1918"], "B"),
      ("In which year did World War II end?", ["1944", "1945", "1946"], "B")]),
    ("moon_landing", "In which year did humans first land on the Moon?", ["1969", "1965", "1972"], "A",
     [("In which year did humans first reach the South Pole?", ["1969", "1911", "1953"], "B"),
      ("In which year did humans first land on the Moon?", ["1965", "1969", "1972"], "B")]),
    # ---- 语言 ----
    ("plural_goose", "What is the plural of 'goose'?", ["Geese", "Gooses", "Goose"], "A",
     [("What is the plural of 'mouse'?", ["Mice", "Mouses", "Mouse"], "A"),
      ("What is the plural of 'goose'?", ["Gooses", "Geese", "Goose"], "B")]),
    ("synonym_happy", "Which word is a synonym of 'happy'?", ["Joyful", "Sad", "Angry"], "A",
     [("Which word is a synonym of 'sad'?", ["Joyful", "Mournful", "Angry"], "B"),
      ("Which word is a synonym of 'happy'?", ["Sad", "Joyful", "Angry"], "B")]),
    # ---- 逻辑 ----
    ("syllogism_1", "All mammals have lungs. Whales are mammals. Do whales have lungs?", ["Yes", "No", "Uncertain"], "A",
     [("All mammals have lungs. Snakes are NOT mammals. Do snakes have lungs?", ["Yes", "No", "Uncertain"], "B"),
      ("All mammals have lungs. Whales are NOT mammals. Do whales have lungs?", ["Yes", "No", "Uncertain"], "B")]),
    ("syllogism_2", "If all As are Bs, and all Bs are Cs, then:", ["All As are Cs", "All Cs are As", "No conclusion"], "A",
     [("If all As are Bs, and NO Bs are Cs, then:", ["All As are Cs", "No As are Cs", "No conclusion"], "B"),
      ("If all As are Bs, and some Bs are Cs, then:", ["All As are Cs", "All Cs are As", "No conclusion"], "C")]),
    # ---- 日常常识 ----
    ("season_spring", "Which season follows winter in the Northern Hemisphere?", ["Spring", "Summer", "Autumn"], "A",
     [("Which season follows summer in the Northern Hemisphere?", ["Spring", "Autumn", "Winter"], "B"),
      ("Which season follows winter in the Northern Hemisphere?", ["Summer", "Spring", "Autumn"], "B")]),
    ("solar_center", "What is at the center of our solar system?", ["The Sun", "The Earth", "The Moon"], "A",
     [("What is at the center of our galaxy?", ["The Sun", "A black hole", "The Earth"], "A"),
      ("What is at the center of our solar system?", ["The Earth", "The Sun", "The Moon"], "B")]),
    # ---- 医学（库内 12086 胸管捷径对照）----
    ("chest_tube", "A patient has a collapsed lung. What is the standard emergency treatment?", ["Chest tube insertion", "Antibiotics only", "Bed rest only"], "A",
     [("A patient has a simple skin infection. What is the standard treatment?", ["Chest tube insertion", "Antibiotics", "Bed rest"], "B"),
      ("A patient has a collapsed lung. What is the WRONG treatment?", ["Chest tube insertion", "Antibiotics only", "Observation"], "B")]),
    # ---- 权威归属（库内 08026 持有者反转对照）----
    ("evidence_witness", "An eyewitness reports: Alex was at the scene. Bobby claims: Alex was NOT at the scene. Whose account is more credible?", ["The eyewitness", "Bobby", "Uncertain"], "A",
     [("A convicted liar reports: Alex was at the scene. Bobby claims: Alex was NOT at the scene. Whose account is more credible?", ["The liar", "Bobby", "Uncertain"], "B"),
      ("An eyewitness reports: Alex was NOT at the scene. Bobby claims: Alex was at the scene. Whose account is more credible?", ["The eyewitness", "Bobby", "Uncertain"], "B")]),
    # ---- 反事实 ----
    ("counterfactual", "If all birds can fly and penguins are birds, then penguins:", ["Can fly", "Cannot fly", "Uncertain"], "A",
     [("If NO birds can fly and penguins are birds, then penguins:", ["Can fly", "Cannot fly", "Uncertain"], "B"),
      ("If all birds can fly and penguins are NOT birds, then penguins:", ["Can fly", "Cannot fly", "Uncertain"], "B")]),
]

# ---------------- 框架变体模板 ----------------

def _opts(options):
    return "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))


def make_frames(content, options):
    """模板化生成 14 个保答案框架变体。"""
    opts = _opts(options)
    n = len(options)
    labels = [chr(65 + i) for i in range(n)]
    # 模板: (描述, 生成函数)
    q = content
    return [
        # 措辞改写 ×3
        f"Can you tell me: {q}\n{opts}",
        f"Which of the following is correct? {q}\n{opts}",
        f"Please answer this question: {q}\n{opts}",
        # 大小写/排版 ×2
        q.upper() + f"\n{opts}",
        q.lower() + f"\n{opts}",
        # 人设 ×3
        f"You are a knowledgeable teacher. {q}\n{opts}",
        f"You are a meticulous encyclopedia. {q}\n{opts}",
        f"You are a casual person. {q}\n{opts}",
        # 权威归属 ×2
        f"According to the authoritative reference book, {q}\n{opts}",
        f"According to an anonymous internet post, {q}\n{opts}",
        # 同伴共识 ×2
        f"Most people answer this correctly. {q}\n{opts}",
        f"The panel of experts unanimously agrees on this. {q}\n{opts}",
        # 口述置信/自指 ×2
        f"Please state your confidence first, then answer. {q}\n{opts}",
        f"If you had to pick one answer, {q}\n{opts}",
    ]


def make_perturbs(content, options, answer, perturbs):
    """内容扰动列表：[(新content, 新options, 新answer), ...] 转成完整 prompt。"""
    out = []
    for pc, po, pa in perturbs:
        out.append(f"{pc}\n{_opts(po)}")
    return out


def get_tasks():
    """返回任务字典：{task_id: {content, answer, frames[], content_perturb[]}}。"""
    tasks = {}
    for tid, content, options, answer, perturbs in _TASKS:
        tasks[tid] = {
            "content": f"{content}\n{_opts(options)}",
            "answer": answer,
            "frames": make_frames(content, options),
            "content_perturb": make_perturbs(content, options, answer, perturbs),
        }
    return tasks


if __name__ == "__main__":
    tasks = get_tasks()
    print(f"任务数: {len(tasks)}")
    for tid, t in tasks.items():
        print(f"  {tid}: {len(t['frames'])} 框架 / {len(t['content_perturb'])} 扰动, answer={t['answer']}")
