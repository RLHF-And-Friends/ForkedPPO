def print_flag_descriptions():
    descriptions = {
        "--n-agents (int)": "число агентов (может быть один)",
        "--comm-matrix-config (str)": "путь до матрицы коммуникаций, задаётся json-ом ([пример](comm_matrices/4_agents.json))",
        "--local-updates (int)": "число циклов локальных обновлений, прежде чем агенты сообщат друг другу об обновлённых весах",
        "--num-envs (int)": "число сред, в которых каждый из агентов учится параллельно",
        "--num-steps (int)": "число шагов, которое агенты делают в каждой из своих сред. Таким образом, в рамках одного \"policy rollout\" мы получаем num_steps * num_envs = batch_size точек в replay buffer для обучения каждого из агентов",
        "--num-minibatches (int)": "число минибатчей, на которые мы делим батч для \"policy rollout\" (см. флаг --num-steps)",
        "--update-epochs (int)": "сколько раз мы просмотрим весь replay buffer целиком во время обучения",
        "--use-comm-penalty (bool)": "добавляем в лосс каждого из агентов сумму из kl-дивергенций с агентами-соседями или нет",
        "--penalty-coeff (float)": "коэффициент регуляризации для суммы kl-дивергенций (см. флаг --use-comm-penalty)",
        "--use-clipping (bool)": "используем клиппинг в лоссе или KL-penalty (a.k.a. adaptive loss)",
        "--clip-coef (float)": "коэффициент регуляризации для клиппинга в функции потерь (см. флаг --use-clipping)",
        "--ent-coef (float)": "коэффициент регуляризации для слагаемого entropy bonus в функции потерь",
        "--vf-coef (float)": "коэффициент регуляризации для слагаемого value function в функции потерь",
        "--anneal-lr": "todo",
        "--gae": "todo",
        "--gamma": "todo",
        "--gae-lambda": "todo",
        "--norm-adv": "todo",
        "--clip-vloss": "todo",
        "--max-grad-norm": "todo",
        "--target-kl": "todo"
    }

    print("\n=== Описание флагов ===\n")
    for flag, description in descriptions.items():
        print(f"{flag}: {description}\n")

if __name__ == "__main__":
    print_flag_descriptions()