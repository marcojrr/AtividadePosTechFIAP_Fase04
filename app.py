"""Interface web do Voice of Customer Intelligence (Streamlit).

Uso, na raiz do projeto e com o venv ativo:
    streamlit run app.py
"""

import html
import warnings
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

from src.embeddings import NILCEmbeddings
from src.llm import criar_llm
from src.pipeline import Resultado, responder
from src.vectorstore import carregar_colecao

st.set_page_config(
    page_title="Voice of Customer",
    page_icon=":material/forum:",
    layout="centered",
)

EXEMPLOS = [
    "O que os clientes elogiam nos produtos?",
    "Quais problemas os clientes relatam com a entrega?",
    "Por que alguns clientes pedem reembolso?",
]

MOTIVOS = {
    "quantitativa": "Pedido de estatística",
    "pergunta_invalida": "Pergunta não reconhecida",
    "fora_do_dominio": "Fora do assunto da base",
    "erro_llm": "Modelo indisponível",
    "falha_fundamentacao": "Resposta sem base suficiente",
    "sem_evidencia_llm": "Sem evidências na base",
    "evidencias_insuficientes": "Poucas evidências",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Newsreader:ital,wght@0,400;0,500;1,400&display=swap');

:root {
  --papel: #F9F9F7;
  --linha: #E6E6E1;
  --tinta: #23262B;
  --suave: #5B616B;
  --acento: #3D4A5C;
  --bolha-usuario: #E4E9F1;
  --borda-ui: #8A9099;
}

.stApp { background: var(--papel); color: var(--tinta); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp textarea,
.stApp button, .stApp summary, .stApp h1, .stApp h2, .stApp h3 {
  font-family: 'DM Sans', system-ui, sans-serif;
}

.block-container { padding-top: 3rem; padding-bottom: 6rem; max-width: 760px; }
footer, [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.titulo {
  font-family: 'Newsreader', Georgia, serif;
  font-size: 2.3rem; font-weight: 500; letter-spacing: -0.01em;
  color: var(--tinta); margin: 0; line-height: 1.15;
}
.subtitulo { color: var(--suave); margin: .4rem 0 2rem; font-size: 1rem; }

/* Balões de chat: pergunta à direita, resposta à esquerda */
[data-testid="stChatMessage"] {
  background: #FFFFFF; border: 1px solid var(--linha); border-radius: 16px;
  padding: .6rem .95rem .4rem;
}
[data-testid^="stChatMessageAvatar"],
[data-testid="stChatMessage"] > :first-child:not([data-testid="stChatMessageContent"]) {
  display: none;
}
[data-testid="stChatMessageContent"] { width: 100%; }
[data-testid="stChatMessage"] :is(p, li, strong, em, summary, h1, h2, h3, h4) {
  color: var(--tinta) !important;
}
[data-testid="stChatMessage"] :is(p, li) { line-height: 1.6; }

div[class*="st-key-usr_"] { align-items: flex-end; }
div[class*="st-key-usr_"] [data-testid="stChatMessage"] {
  width: fit-content; max-width: 82%;
  background: var(--bolha-usuario); border-color: #CDD5E3; border-top-right-radius: 4px;
}
div[class*="st-key-bot_"] { align-items: flex-start; }
div[class*="st-key-bot_"] [data-testid="stChatMessage"] {
  width: 94%; border-top-left-radius: 4px;
}

.hora { text-align: right; color: var(--suave); font-size: .75rem; margin-top: .15rem; }
.dica { color: var(--suave); font-size: .95rem; margin: 0 0 .6rem; }
.selo {
  display: inline-block; font-size: .8rem; color: var(--acento);
  background: #E9ECF0; border-radius: 999px; padding: .15rem .7rem; margin-bottom: .5rem;
}
.secao { color: var(--suave); font-size: .85rem; margin: 1.2rem 0 .4rem; }

[data-testid="stExpander"] details {
  background: #F7F7F4; border: 1px solid var(--linha); border-radius: 12px;
}
.citacao {
  font-family: 'Newsreader', Georgia, serif; font-size: 1.08rem; line-height: 1.55;
  color: var(--tinta); border-left: 2px solid var(--linha);
  padding-left: 1rem; margin: .2rem 0 .7rem;
}
.meta { color: var(--suave); font-size: .85rem; line-height: 1.5; }

.stButton > button {
  background: #FFFFFF; border: 1px solid var(--borda-ui); border-radius: 12px;
  justify-content: flex-start; text-align: left; min-height: 2.75rem;
}
.stButton > button p { color: var(--tinta); }
.stButton > button:hover { border-color: var(--acento); }
.stButton > button:hover p { color: var(--acento); }

[data-testid="stSidebar"] { background: #F1F1ED; }
[data-testid="stSidebar"] :is(p, label) { color: var(--tinta); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--suave) !important; }

[data-testid="stBottom"], [data-testid="stBottom"] > div { background: var(--papel); }
[data-testid="stChatInput"] { background: #FFFFFF; border: 1px solid var(--borda-ui); border-radius: 14px; }
[data-testid="stChatInput"] textarea { background: #FFFFFF; color: var(--tinta); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--suave); opacity: 1; }

button:focus-visible, summary:focus-visible, [data-testid="stChatInput"]:focus-within {
  outline: 2px solid var(--acento); outline-offset: 2px;
}

@media (max-width: 640px) {
  .block-container { padding-top: 1.5rem; }
  .titulo { font-size: 1.8rem; }
  div[class*="st-key-usr_"] [data-testid="stChatMessage"] { max-width: 92%; }
  div[class*="st-key-bot_"] [data-testid="stChatMessage"] { width: 100%; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sistema (carregado uma única vez)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Carregando o modelo e o banco vetorial (só na primeira vez)...")
def carregar_sistema():
    store = carregar_colecao(NILCEmbeddings())
    if not store.get(limit=1)["ids"]:
        raise RuntimeError("O índice vetorial está vazio. Rode: python -m src.vectorstore")
    return store, criar_llm()


# --------------------------------------------------------------------------- #
# Apoio de exibição
# --------------------------------------------------------------------------- #
def _md(texto: str) -> str:
    """Evita que '$' em avaliações (ex.: R$ 50) seja lido como fórmula."""
    return texto.replace("$", "\\$")


def _html(texto: str) -> str:
    return html.escape(texto).replace("$", "&#36;")


def _previa(texto: str, limite: int = 60) -> str:
    texto = " ".join(texto.split())
    return texto if len(texto) <= limite else texto[:limite].rstrip() + "…"


def _notas(notas: list[int]) -> str:
    contagem = sorted(Counter(notas).items(), reverse=True)
    return "Notas: " + ", ".join(f"{n} ({c}×)" for n, c in contagem)


def mostrar_resultado(r: Resultado, detalhes: bool) -> None:
    res = r.resposta

    if r.motivo:
        rotulo = MOTIVOS.get(r.motivo, "Sem resposta")
        st.markdown(f'<span class="selo">{rotulo}</span>', unsafe_allow_html=True)
    st.markdown(_md(res.texto))

    if res.citadas:
        st.markdown('<p class="secao">Avaliações que sustentam a resposta</p>', unsafe_allow_html=True)
        for id_ev, ev in res.citadas.items():
            n = ev.ocorrencias
            rotulo_ev = f"[{id_ev}] {n} {'avaliação' if n == 1 else 'avaliações'}: “{_md(_previa(ev.texto))}”"
            with st.expander(rotulo_ev):
                st.markdown(f'<div class="citacao">{_html(ev.texto)}</div>', unsafe_allow_html=True)
                ids = ", ".join(ev.review_ids[:3])
                if len(ev.review_ids) > 3:
                    ids += f" e mais {len(ev.review_ids) - 3}"
                st.markdown(
                    f'<div class="meta">{_notas(ev.notas)}<br>review_id: {_html(ids)}</div>',
                    unsafe_allow_html=True,
                )

    if detalhes:
        with st.expander("Detalhes técnicos"):
            if r.similaridade_pergunta is not None:
                st.markdown(f"Similaridade da pergunta com a base: **{r.similaridade_pergunta:.3f}**")
            if r.consultas:
                st.markdown("Buscas geradas pelo modelo:")
                for c in r.consultas:
                    st.markdown(f"- {_md(c)}")
            if res.nao_citadas:
                st.markdown("Recuperadas e não citadas:")
                for id_ev, ev in res.nao_citadas.items():
                    st.markdown(f"- [{id_ev}] {_md(ev.texto)}")


def _usar_exemplo(texto: str) -> None:
    st.session_state.pendente = texto


def _limpar() -> None:
    st.session_state.mensagens = []


# --------------------------------------------------------------------------- #
# Tela
# --------------------------------------------------------------------------- #
st.session_state.setdefault("mensagens", [])
st.session_state.setdefault("pendente", None)

with st.sidebar:
    detalhes = st.toggle(
        "Mostrar detalhes técnicos",
        help="Buscas geradas pelo modelo, similaridade e evidências não citadas.",
    )
    with st.popover("Limpar conversa", width="stretch"):
        st.write("Apagar todas as mensagens desta conversa?")
        st.button("Sim, apagar", on_click=_limpar, type="primary")
    st.caption(
        "As respostas vêm de avaliações reais de clientes da Olist. "
        "Perguntas que a base não consegue sustentar são recusadas."
    )

st.markdown(
    '<h1 class="titulo">Voice of Customer</h1>'
    '<p class="subtitulo">Pergunte em linguagem natural o que os clientes dizem nas avaliações da Olist.</p>',
    unsafe_allow_html=True,
)

try:
    store, llm = carregar_sistema()
except Exception as erro:
    st.error(
        f"Não foi possível carregar o sistema ({type(erro).__name__}): {erro}\n\n"
        "Confira o arquivo .env e se o índice foi criado com `python -m src.vectorstore`."
    )
    st.stop()

for i, msg in enumerate(st.session_state.mensagens):
    if msg["papel"] == "user":
        with st.container(key=f"usr_{i}"), st.chat_message("user"):
            st.markdown(_md(msg["texto"]))
            st.markdown(f'<div class="hora">{msg["hora"]}</div>', unsafe_allow_html=True)
    else:
        with st.container(key=f"bot_{i}"), st.chat_message("assistant"):
            mostrar_resultado(msg["resultado"], detalhes)
            st.markdown(f'<div class="hora">{msg["hora"]}</div>', unsafe_allow_html=True)

area_exemplos = st.empty()
if not st.session_state.mensagens:
    with area_exemplos.container():
        st.markdown('<p class="dica">Experimente uma destas perguntas:</p>', unsafe_allow_html=True)
        for exemplo in EXEMPLOS:
            st.button(exemplo, key=f"ex_{exemplo}", on_click=_usar_exemplo, args=(exemplo,), width="stretch")

pergunta = st.chat_input("Pergunte algo sobre as avaliações dos clientes")
if not pergunta and st.session_state.pendente:
    pergunta = st.session_state.pendente
st.session_state.pendente = None

if pergunta:
    area_exemplos.empty()
    n = len(st.session_state.mensagens)
    hora_pergunta = datetime.now().strftime("%H:%M")
    with st.container(key=f"usr_{n}"), st.chat_message("user"):
        st.markdown(_md(pergunta))
        st.markdown(f'<div class="hora">{hora_pergunta}</div>', unsafe_allow_html=True)
    with st.container(key=f"bot_{n + 1}"), st.chat_message("assistant"):
        try:
            with st.spinner("Buscando avaliações e escrevendo a resposta. Leva cerca de 30 segundos."):
                resultado = responder(pergunta, store, llm)
        except Exception as erro:
            st.error("Não consegui responder agora. Tente de novo em instantes; se persistir, confira a cota da chave do Gemini.")
            st.caption(f"Detalhe técnico: {type(erro).__name__}: {erro}")
            st.stop()
        hora_resposta = datetime.now().strftime("%H:%M")
        mostrar_resultado(resultado, detalhes)
        st.markdown(f'<div class="hora">{hora_resposta}</div>', unsafe_allow_html=True)
    st.session_state.mensagens.append({"papel": "user", "texto": pergunta, "hora": hora_pergunta})
    st.session_state.mensagens.append({"papel": "assistant", "resultado": resultado, "hora": hora_resposta})