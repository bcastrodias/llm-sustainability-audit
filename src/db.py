import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config import DB_PATH, _assign_pole


SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id                  TEXT PRIMARY KEY,   -- e.g. Q01
    theme               TEXT NOT NULL,
    dimension           TEXT NOT NULL,
    source              TEXT NOT NULL,      -- original | added
    incremental_is_a    INTEGER NOT NULL,   -- 1 = incremental presented as A, 0 = as B
    pole_a_en           TEXT NOT NULL,
    pole_b_en           TEXT NOT NULL,
    pole_a_pt           TEXT NOT NULL,
    pole_b_pt           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    name        TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    api_base    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id     TEXT NOT NULL REFERENCES questions(id),
    model_name      TEXT NOT NULL REFERENCES models(name),
    order_ab        TEXT NOT NULL CHECK(order_ab IN ('AB', 'BA')),
    persona         TEXT NOT NULL,
    language        TEXT NOT NULL CHECK(language IN ('en', 'pt')),
    temperature     REAL NOT NULL,
    rep_number      INTEGER NOT NULL,
    choice          TEXT CHECK(choice IN ('A', 'B', 'refused', 'error')),
    justification   TEXT,
    raw_response    TEXT,
    tokens_input    INTEGER,
    tokens_output   INTEGER,
    cost_usd        REAL,
    latency_ms      INTEGER,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(question_id, model_name, order_ab, persona, language, temperature, rep_number)
);

CREATE TABLE IF NOT EXISTS api_health (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name  TEXT NOT NULL,
    checked_at  TEXT DEFAULT (datetime('now')),
    status      TEXT NOT NULL CHECK(status IN ('ok', 'error')),
    latency_ms  INTEGER,
    error_msg   TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_question  ON runs(question_id);
CREATE INDEX IF NOT EXISTS idx_runs_model     ON runs(model_name);
CREATE INDEX IF NOT EXISTS idx_runs_choice    ON runs(choice);
"""

QUESTIONS_DATA = [
    ("Q01", "Ecological transition mechanism", "Market vs. State", "original",
     "The view that market-based instruments — such as carbon pricing — are the most effective mechanisms for promoting ecological transition by enabling agents to conduct rational cost-benefit analyses.",
     "The view that a successful transition requires state-led arrangements, including regulation, centralized planning, and mission-oriented industrial and fiscal policies.",
     "A visão de que instrumentos de mercado — como a precificação de carbono — são os mecanismos mais eficazes para promover a transição ecológica, permitindo que os agentes realizem análises racionais de custo-benefício.",
     "A visão de que uma transição bem-sucedida requer arranjos liderados pelo Estado, incluindo regulação, planejamento centralizado e políticas industriais e fiscais orientadas por missão."),

    ("Q02", "Externalities internalization", "Market vs. State", "original",
     "The view that the establishment of well-defined property rights is an effective mechanism for internalizing externalities and fostering environmental protection.",
     "The view that explicit command-and-control policies, such as the enforcement of strict conservation areas, are more effective for objectives such as preserving standing forests and preventing biodiversity loss.",
     "A visão de que o estabelecimento de direitos de propriedade bem definidos é um mecanismo eficaz para internalizar externalidades e promover a proteção ambiental.",
     "A visão de que políticas explícitas de comando e controle, como a aplicação de áreas de conservação rigorosas, são mais eficazes para objetivos como preservar florestas em pé e prevenir a perda de biodiversidade."),

    ("Q03", "Ecosystem services conservation", "Efficiency vs. Structural Transformation", "original",
     "The view that establishing financial incentives — such as Payments for Ecosystem Services (PES) and the Tropical Forest Forever Fund — is the most effective policy for rewarding agents for the conservation of ecosystem services.",
     "The view that empowering local and traditional communities to develop non-extractive, bioeconomic initiatives is a fairer and more effective strategy for tackling environmental degradation.",
     "A visão de que o estabelecimento de incentivos financeiros — como Pagamentos por Serviços Ecossistêmicos (PSE) e o Fundo Floresta Tropical para Sempre — é a política mais eficaz para recompensar agentes pela conservação dos serviços ecossistêmicos.",
     "A visão de que empoderar comunidades locais e tradicionais para desenvolver iniciativas bioeconômicas não extrativistas é uma estratégia mais justa e eficaz para combater a degradação ambiental."),

    ("Q04", "Scope of transition", "Efficiency vs. Structural Transformation", "original",
     "The view that the primary goal of ecological transition policies should be to promote a stable transition toward a sustainable energy matrix and improved resource efficiency.",
     "The view that policy must aim for broader structural transformations, including deliberate shifts in priority activities, production modes, and societal consumption patterns.",
     "A visão de que o objetivo principal das políticas de transição ecológica deve ser promover uma transição estável para uma matriz energética sustentável e maior eficiência no uso de recursos.",
     "A visão de que a política deve visar transformações estruturais mais amplas, incluindo mudanças deliberadas nas atividades prioritárias, nos modos de produção e nos padrões de consumo da sociedade."),

    ("Q05", "Policy design neutrality", "Efficiency vs. Structural Transformation", "original",
     "The view that policy design should prioritize neutrality — maintaining price uniformity and avoiding sectoral favoritism to ensure market efficiency.",
     "The view that policy must be deliberately interventionist, actively incentivizing certain resource uses while penalizing others based on their social and biophysical impact.",
     "A visão de que o desenho de políticas deve priorizar a neutralidade — mantendo a uniformidade de preços e evitando o favorecimento setorial para garantir a eficiência do mercado.",
     "A visão de que a política deve ser deliberadamente intervencionista, incentivando ativamente certos usos de recursos e penalizando outros com base em seu impacto social e biofísico."),

    ("Q06", "Speed of policy action", "Gradualism vs. Radical Urgency", "original",
     "The view that policies such as carbon taxes should be phased in gradually to allow economic agents to adapt while maintaining a trajectory toward increasingly stringent emission reductions.",
     "The view that the imminent risks of climate tipping points and the transgression of planetary boundaries necessitate immediate, substantial policy interventions that prioritize systemic change over incremental adjustment.",
     "A visão de que políticas como impostos sobre carbono devem ser introduzidas gradualmente para permitir que os agentes econômicos se adaptem, mantendo uma trajetória em direção a reduções de emissões cada vez mais rigorosas.",
     "A visão de que os riscos iminentes de pontos de inflexão climáticos e a transgressão dos limites planetários necessitam de intervenções políticas imediatas e substanciais que priorizem a mudança sistêmica em detrimento do ajuste incremental."),

    ("Q07", "Jevons paradox", "Efficiency vs. Structural Transformation", "original",
     "The view that the rebound effect (Jevons paradox) is primarily a short-term adjustment challenge that can be overcome by further driving efficiency gains alongside taxes on residual consumption.",
     "The view that efficiency gains are fundamentally incapable of ensuring resource conservation, making the establishment of absolute physical limits and resource caps the necessary policy response.",
     "A visão de que o efeito rebote (paradoxo de Jevons) é principalmente um desafio de ajuste de curto prazo que pode ser superado impulsionando ainda mais os ganhos de eficiência junto com impostos sobre o consumo residual.",
     "A visão de que os ganhos de eficiência são fundamentalmente incapazes de garantir a conservação dos recursos, tornando o estabelecimento de limites físicos absolutos e tetos de recursos a resposta política necessária."),

    ("Q08", "Central bank role", "Gradualism vs. Radical Urgency", "original",
     "The view that while central banks and governments can assist by structuring taxonomies to reduce information asymmetries and guide green investment, monetary policy should primarily maintain its restrictive focus on price and financial system stability.",
     "The view that monetary authorities must acknowledge their inherent lack of neutrality and proactively prioritize — by means of mandate changes, credit steering, and precautionary prudential supervision — ecological transition.",
     "A visão de que, embora bancos centrais e governos possam auxiliar estruturando taxonomias para reduzir assimetrias de informação e orientar o investimento verde, a política monetária deve manter principalmente seu foco restritivo na estabilidade de preços e do sistema financeiro.",
     "A visão de que as autoridades monetárias devem reconhecer sua falta inerente de neutralidade e priorizar proativamente — por meio de mudanças de mandato, direcionamento de crédito e supervisão prudencial precautória — a transição ecológica."),

    ("Q09", "Policy instrument mix", "Market vs. State", "original",
     "The view that market-based incentives and monetary policies are preferable for their efficiency, lower administrative costs, and non-distortive impact on economic behavior.",
     "The view that robust fiscal policies and regulatory mandates are necessary to achieve the rapid, systemic changes required for ecological transition.",
     "A visão de que incentivos de mercado e políticas monetárias são preferíveis por sua eficiência, menores custos administrativos e impacto não distorsivo no comportamento econômico.",
     "A visão de que políticas fiscais robustas e mandatos regulatórios são necessários para alcançar as mudanças rápidas e sistêmicas exigidas para a transição ecológica."),

    ("Q10", "Global climate governance", "Efficiency vs. Climate Justice", "original",
     "The view that international institutions should shift toward an incentive structure that internalizes the costs and benefits of emission reduction goals — such as the Climate Clubs proposed by William Nordhaus.",
     "The view that global climate governance must move away from trade-linked coercion toward frameworks based on the principle of common but differentiated responsibilities.",
     "A visão de que as instituições internacionais devem se orientar para uma estrutura de incentivos que internalize os custos e benefícios das metas de redução de emissões — como os Clubes do Clima propostos por William Nordhaus.",
     "A visão de que a governança climática global deve se afastar da coerção vinculada ao comércio em direção a marcos baseados no princípio das responsabilidades comuns, porém diferenciadas."),

    ("Q11", "Historical climate debt", "Efficiency vs. Climate Justice", "original",
     "The view that while targeted compensation is necessary, global climate governance should primarily rely on horizontal, shared responsibility frameworks.",
     "The view that climate action must fundamentally address historical injustice, necessitating that international institutions provide structural, reparative solutions — such as technology transfer, low-cost financing, and external debt cancellation — to facilitate the transition and development of the Global South.",
     "A visão de que, embora uma compensação direcionada seja necessária, a governança climática global deve se basear principalmente em marcos horizontais de responsabilidade compartilhada.",
     "A visão de que a ação climática deve fundamentalmente abordar a injustiça histórica, exigindo que as instituições internacionais forneçam soluções estruturais e reparadoras — como transferência de tecnologia, financiamento de baixo custo e cancelamento da dívida externa — para facilitar a transição e o desenvolvimento do Sul Global."),

    ("Q12", "Knowledge systems in policy", "Efficiency vs. Climate Justice", "added",
     "The view that environmental policy design should be primarily guided by quantitative cost-benefit analyses and formal economic modelling.",
     "The view that plural knowledges — including indigenous, community-based, and non-Western ecological knowledge systems — should be given equal standing alongside technical expertise in policy deliberation.",
     "A visão de que o desenho de políticas ambientais deve ser guiado principalmente por análises quantitativas de custo-benefício e modelagem econômica formal.",
     "A visão de que conhecimentos plurais — incluindo sistemas de conhecimento ecológico indígenas, comunitários e não ocidentais — devem ter igual peso ao lado da expertise técnica na deliberação política."),

    ("Q13", "Growth paradigm", "Efficiency vs. Structural Transformation", "added",
     "The view that decoupling economic growth from resource use and emissions — through green innovation and circular economy — is both feasible and desirable as the foundation of ecological transition.",
     "The view that the biophysical limits of the planet make continued GDP growth fundamentally incompatible with genuine ecological sustainability, requiring post-growth or degrowth economic frameworks.",
     "A visão de que desacoplar o crescimento econômico do uso de recursos e das emissões — por meio de inovação verde e economia circular — é viável e desejável como base da transição ecológica.",
     "A visão de que os limites biofísicos do planeta tornam o crescimento contínuo do PIB fundamentalmente incompatível com a sustentabilidade ecológica genuína, exigindo marcos econômicos de pós-crescimento ou decrescimento."),

    ("Q14", "Valuation of nature", "Efficiency vs. Climate Justice", "added",
     "The view that assigning economic values to ecosystem services and biodiversity — through mechanisms such as natural capital accounting and biodiversity offsets — is the most effective way to integrate nature into decision-making.",
     "The view that the commodification of nature through such frameworks systematically undervalues non-quantifiable ecological relationships and undermines the intrinsic rights of nature.",
     "A visão de que atribuir valores econômicos aos serviços ecossistêmicos e à biodiversidade — por meio de mecanismos como contabilidade do capital natural e compensações de biodiversidade — é a forma mais eficaz de integrar a natureza na tomada de decisões.",
     "A visão de que a mercantilização da natureza por meio desses marcos subestima sistematicamente as relações ecológicas não quantificáveis e mina os direitos intrínsecos da natureza."),
]


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _seed_questions(conn)


def _seed_questions(conn: sqlite3.Connection) -> None:
    existing = {row["id"] for row in conn.execute("SELECT id FROM questions")}
    for row in QUESTIONS_DATA:
        qid = row[0]
        if qid not in existing:
            incremental_is_a = int(_assign_pole(qid))
            # Insert: id, theme, dimension, source, incremental_is_a, pole_a_en, pole_b_en, pole_a_pt, pole_b_pt
            # QUESTIONS_DATA stores incremental text first — swap if incremental_is_a=False
            id_, theme, dimension, source, ms_en, het_en, ms_pt, het_pt = row
            if incremental_is_a:
                pole_a_en, pole_b_en = ms_en, het_en
                pole_a_pt, pole_b_pt = ms_pt, het_pt
            else:
                pole_a_en, pole_b_en = het_en, ms_en
                pole_a_pt, pole_b_pt = het_pt, ms_pt
            conn.execute(
                "INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?)",
                (id_, theme, dimension, source, incremental_is_a, pole_a_en, pole_b_en, pole_a_pt, pole_b_pt),
            )


def seed_models(models) -> None:
    with get_conn() as conn:
        for m in models:
            conn.execute(
                "INSERT OR REPLACE INTO models VALUES (?,?,?,?)",
                (m.name, m.provider, m.model_id, m.api_base),
            )


def save_run(run: dict) -> None:
    cols = (
        "question_id", "model_name", "order_ab", "persona", "language",
        "temperature", "rep_number", "choice", "justification", "raw_response",
        "tokens_input", "tokens_output", "cost_usd", "latency_ms",
    )
    placeholders = ",".join("?" * len(cols))
    with get_conn() as conn:
        # OR REPLACE: overwrites previous error rows so retries on resume work correctly.
        # Successful runs are protected by the UNIQUE constraint — a completed rep
        # will simply be replaced with identical data, which is harmless.
        conn.execute(
            f"INSERT OR REPLACE INTO runs ({','.join(cols)}) VALUES ({placeholders})",
            [run.get(c) for c in cols],
        )


def save_health(model_name: str, status: str, latency_ms: int | None, error_msg: str | None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO api_health (model_name, status, latency_ms, error_msg) VALUES (?,?,?,?)",
            (model_name, status, latency_ms, error_msg),
        )


def count_runs(question_id: str, model_name: str, order_ab: str,
               persona: str, language: str, temperature: float) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT COUNT(*) FROM runs
               WHERE question_id=? AND model_name=? AND order_ab=?
               AND persona=? AND language=? AND temperature=?
               AND choice NOT IN ('error')""",
            (question_id, model_name, order_ab, persona, language, temperature),
        ).fetchone()
        return row[0]


def get_all_questions() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM questions ORDER BY id").fetchall()
