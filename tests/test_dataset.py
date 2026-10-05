from nimbus_llm.dataset import SYSTEM_PROMPT, build_dataset, read_jsonl
from nimbus_llm.config import repo_path
from nimbus_llm.scoring import pairwise_win, score_example


def test_dataset_splits_are_disjoint_and_schema_valid(tmp_path):
    counts = build_dataset(repo_path("data/raw/product_facts.yaml"), tmp_path)
    assert counts["train"] > counts["val"] > 0
    assert counts["test"] > 0
    users = set()
    for split in ("train", "val", "test"):
        rows = read_jsonl(tmp_path / f"{split}.jsonl")
        assert rows
        for row in rows:
            assert row["split"] == split
            assert row["messages"][0]["role"] == "system"
            assert row["messages"][0]["content"] == SYSTEM_PROMPT
            assert row["messages"][1]["role"] == "user"
            assert row["messages"][-1]["role"] == "assistant"
            user = row["messages"][1]["content"].strip().lower()
            assert user not in users
            users.add(user)
            assert isinstance(row["must_include"], list)


def test_fact_coverage_scoring():
    row = {
        "must_include": ["600", "Pro"],
        "must_not_include": ["NMB-999"],
        "id": "x",
        "topic": "rate_limits",
        "fact_id": "rate.pro",
    }
    good = score_example("The Pro plan is limited to 600 requests per minute.", row)
    assert good["pass"] is True
    bad = score_example("Unlimited RPM, see NMB-999.", row)
    assert bad["pass"] is False
    assert pairwise_win(bad, good) == "adapter"
