from methods.all_history import AllHistory
from methods.autogen_gmemory_online import AutoGenGMemoryOnline
from methods.awm_online import AwmOnline
from methods.baseline import Baseline
from methods.dynamic_cheatsheet_retrieval_synthesis import DynamicCheatsheetRetrievalSynthesis
from methods.expel_online_mt import ExpelOnlineMT
from methods.expel_online_st import ExpelOnlineST
from methods.exp_recent import ExpRecent
from methods.history_rag import HistoryRAG

METHOD_REGISTRY = {
    AllHistory.name: AllHistory,
    AutoGenGMemoryOnline.name: AutoGenGMemoryOnline,
    "AutoGenGMemory": AutoGenGMemoryOnline,
    "g-memory-autogen": AutoGenGMemoryOnline,
    AwmOnline.name: AwmOnline,
    "AWMOnline": AwmOnline,
    Baseline.name: Baseline,
    "NoMemory": Baseline,
    DynamicCheatsheetRetrievalSynthesis.name: DynamicCheatsheetRetrievalSynthesis,
    "DynamicCheatsheet_RetrievalSynthesis": DynamicCheatsheetRetrievalSynthesis,
    ExpelOnlineMT.name: ExpelOnlineMT,
    "ExpeL-Online": ExpelOnlineMT,
    "ExpeLOnline": ExpelOnlineMT,
    # max-tries=1 variant: same class, separate registry name so holdout
    # outputs land in a different subdir (.../ExpeL-Online-MT-tries1/) and
    # don't clobber the canonical max-tries=3 results.
    "ExpeL-Online-MT-tries1": ExpelOnlineMT,
    ExpelOnlineST.name: ExpelOnlineST,
    "ExpeL-Online-ST": ExpelOnlineST,
    "ExpeLOnlineST": ExpelOnlineST,
    ExpRecent.name: ExpRecent,
    HistoryRAG.name: HistoryRAG,
    "ExpRAG": HistoryRAG,
}

__all__ = [
    "AllHistory",
    "AutoGenGMemoryOnline",
    "AwmOnline",
    "Baseline",
    "DynamicCheatsheetRetrievalSynthesis",
    "ExpelOnlineMT",
    "ExpelOnlineST",
    "ExpRecent",
    "HistoryRAG",
    "METHOD_REGISTRY",
]
