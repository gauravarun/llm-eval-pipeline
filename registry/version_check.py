# registry/version_check.py
# Run this standalone to inspect registered dataset versions.

from dataset_store import register_dataset, list_versions, get_tasks, TASKS

if __name__ == "__main__":
    version = register_dataset(TASKS)
    versions = list_versions()

    print("\n  Dataset Registry")
    print(f"  {'─'*40}")
    for v in versions:
        print(f"  {v['version_id']}  |  {v['num_tasks']} tasks  |  checksum: {v['checksum']}  |  {v['registered_at']}")
        print(f"         categories: {', '.join(v['categories'])}")
    print(f"\n  Current: {version['version_id']} ({version['checksum']})")
    print(f"  Tasks loaded: {len(get_tasks())}")
    print(f"  Easy: {len(get_tasks('easy'))}  Medium: {len(get_tasks('medium'))}  Hard: {len(get_tasks('hard'))}\n")