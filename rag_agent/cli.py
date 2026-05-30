import argparse
import sys

from dotenv import load_dotenv

from .pipeline import RAGPipeline, RAGConfig, AgentRAGPipeline, DB_DIR_DEFAULT, DOCS_DIR_DEFAULT
from . import logger as log
from .evaluation import TestSuite, RAGEvaluator, EvalReporter

load_dotenv()


def _add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")


def _add_agent_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", action="store_true", help="使用 LangGraph Agent 多步推理模式")


def _make_pipeline(args, config):
    """根据 --agent 标志创建对应管线。"""
    if getattr(args, "agent", False):
        return AgentRAGPipeline(db_dir=args.db_dir, config=config)
    return RAGPipeline(db_dir=args.db_dir, config=config)


def _add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=30, help="候选检索数量 (默认30)")
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid",
                        help="检索模式: vector(语义) / bm25(关键词) / hybrid(混合)")
    parser.add_argument("--no-rerank", action="store_true", help="禁用 LLM 精排")
    parser.add_argument("--rerank-top-k", type=int, default=8, help="精排后保留数量 (默认8)")
    parser.add_argument("--no-rewrite", action="store_true", help="禁用查询改写")
    parser.add_argument("--no-hyde", action="store_true", help="禁用 HyDE 假设文档检索")


def cmd_ingest(args):
    log.setup(verbose=args.verbose)
    config = RAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_strategy=args.chunk_strategy,
        collection_name=args.collection,
    )
    pipeline = RAGPipeline(db_dir=args.db_dir, config=config)
    count = pipeline.ingest(docs_dir=args.docs_dir)
    if count == 0:
        sys.exit(1)


def cmd_query(args):
    log.setup(verbose=args.verbose)
    config = RAGConfig(
        top_k=args.top_k,
        collection_name=args.collection,
        retrieval_mode=args.mode,
        enable_rerank=not args.no_rerank,
        rerank_top_k=args.rerank_top_k,
        enable_query_rewrite=not args.no_rewrite,
        enable_hyde=not args.no_hyde,
    )
    if getattr(args, "agent", False):
        print("正在初始化 Agent ...", end=" ", flush=True)
    pipeline = _make_pipeline(args, config)
    if getattr(args, "agent", False):
        print("就绪")
    result = pipeline.query(args.question)
    print(f"问题: {result['question']}")
    print(f"回答: {result['answer']}")
    print(f"来源: {', '.join(result['sources'])}")


def cmd_interactive(args):
    log.setup(verbose=args.verbose)
    config = RAGConfig(
        top_k=args.top_k,
        collection_name=args.collection,
        retrieval_mode=args.mode,
        enable_rerank=not args.no_rerank,
        rerank_top_k=args.rerank_top_k,
        enable_query_rewrite=not args.no_rewrite,
        enable_hyde=not args.no_hyde,
    )
    pipeline = _make_pipeline(args, config)
    mode_label = "Agent" if getattr(args, "agent", False) else "Pipeline"
    print(f"RAG 交互问答 ({mode_label} 模式, 输入 'exit' 退出)")
    print(f"检索: {args.mode} | Rerank: {not args.no_rerank} | "
          f"查询改写: {not args.no_rewrite} | HyDE: {not args.no_hyde}")
    print("-" * 50)
    while True:
        try:
            q = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        result = pipeline.query(q)
        print(f"\n{result['answer']}")
        if args.show_sources:
            print(f"\n来源: {', '.join(result['sources'])}")


def cmd_evaluate(args):
    """运行 RAG 效果评估"""
    log.setup(verbose=args.verbose)

    # 加载测试用例
    if args.categories:
        suite = TestSuite(name="custom")
        all_cases = TestSuite.load_default()
        for cat in args.categories.split(","):
            cat = cat.strip()
            for tc in all_cases.get_by_category(cat):
                suite.add_case(tc)
        if len(suite) == 0:
            print(f"未找到匹配类别的测试用例: {args.categories}")
            print(f"可用类别: {', '.join(all_cases.categories)}")
            sys.exit(1)
    else:
        suite = TestSuite.load_default()

    # 初始化管线
    config = RAGConfig(
        top_k=args.top_k,
        collection_name=args.collection,
        retrieval_mode=args.mode,
        enable_rerank=not args.no_rerank,
        rerank_top_k=args.rerank_top_k,
        enable_query_rewrite=not args.no_rewrite,
        enable_hyde=not args.no_hyde if hasattr(args, 'no_hyde') else True,
    )
    pipeline = _make_pipeline(args, config)

    # 确定检索模式
    if args.eval_modes:
        modes = [m.strip() for m in args.eval_modes.split(",")]
    else:
        modes = ["vector", "bm25", "hybrid", "hybrid_reranked"]

    # 确定 K 值
    if args.k_values:
        k_values = [int(k.strip()) for k in args.k_values.split(",")]
    else:
        k_values = [1, 3, 5, 8]

    # 运行评估
    evaluator = RAGEvaluator(pipeline, suite)

    run_retrieval = not args.generation_only
    run_generation = not args.retrieval_only

    retrieval_summary = {}
    generation_summary = {}
    per_category = {}

    if run_retrieval:
        retrieval_summary = evaluator.evaluate_retrieval(
            modes=modes, k_values=k_values, top_k=args.eval_top_k, verbose=not args.quiet
        )
        if args.by_category:
            per_category = evaluator.evaluate_by_category(
                modes=modes, k_values=k_values, top_k=args.eval_top_k, verbose=not args.quiet
            )

    if run_generation:
        generation_summary = evaluator.evaluate_generation(verbose=not args.quiet)

    # 保存逐用例详情（默认按时间戳自动命名）
    if not args.no_save_details:
        details_path = args.save_details
        if not details_path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            details_path = f"benchmarks/details_{timestamp}.json"
        evaluator.save_details(details_path)

    # 报告
    from .evaluation.runner import EvalResult
    result = EvalResult(
        test_suite_name=suite.name,
        retrieval=retrieval_summary,
        generation=generation_summary,
        per_category=per_category,
    )

    EvalReporter.print_full_report(result, show_per_category=args.by_category)

    if args.output:
        EvalReporter.export_json(result, args.output)


def main():
    parser = argparse.ArgumentParser(description="RAG Agent - 基于检索增强生成的问答系统")
    sub = parser.add_subparsers(dest="command", help="子命令")

    # ingest
    p_ingest = sub.add_parser("ingest", help="导入文档到向量数据库")
    p_ingest.add_argument("--docs-dir", default=DOCS_DIR_DEFAULT, help="文档目录")
    p_ingest.add_argument("--db-dir", default=DB_DIR_DEFAULT, help="向量数据库目录")
    p_ingest.add_argument("--chunk-size", type=int, default=1200,
                          help="段落目标大小（默认1200字）")
    p_ingest.add_argument("--chunk-overlap", type=int, default=25)
    p_ingest.add_argument("--chunk-strategy",
                          choices=["paragraph", "sentence", "recursive"],
                          default="paragraph",
                          help="切分策略: paragraph(段落)/sentence(句子)/recursive(递归)")
    p_ingest.add_argument("--collection", default="rag_agent")
    _add_verbose_arg(p_ingest)

    # query
    p_query = sub.add_parser("query", help="单次问答")
    p_query.add_argument("question", help="问题")
    p_query.add_argument("--db-dir", default=DB_DIR_DEFAULT)
    p_query.add_argument("--collection", default="rag_agent")
    _add_verbose_arg(p_query)
    _add_agent_arg(p_query)
    _add_retrieval_args(p_query)

    # interactive
    p_chat = sub.add_parser("chat", help="交互式问答")
    p_chat.add_argument("--db-dir", default=DB_DIR_DEFAULT)
    p_chat.add_argument("--collection", default="rag_agent")
    p_chat.add_argument("--show-sources", action="store_true")
    _add_verbose_arg(p_chat)
    _add_agent_arg(p_chat)
    _add_retrieval_args(p_chat)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="RAG 效果评估")
    p_eval.add_argument("--db-dir", default=DB_DIR_DEFAULT)
    p_eval.add_argument("--collection", default="rag_agent")
    p_eval.add_argument("--categories", default=None,
                        help="按类别筛选测试用例，逗号分隔 (如: definition,comparison)")
    p_eval.add_argument("--retrieval-only", action="store_true", help="仅运行检索评估")
    p_eval.add_argument("--generation-only", action="store_true", help="仅运行生成评估")
    p_eval.add_argument("--eval-modes", default=None,
                        help="检索模式，逗号分隔 (如: vector,bm25,hybrid)，默认全部")
    p_eval.add_argument("--eval-top-k", type=int, default=20,
                        help="评估时每种模式检索的文档数 (默认20)")
    p_eval.add_argument("--k-values", default=None,
                        help="Recall@K 中的 K 值列表，逗号分隔 (如: 1,3,5,8)")
    p_eval.add_argument("--by-category", action="store_true", help="按类别拆分指标")
    p_eval.add_argument("--output", default=None, help="导出 JSON 报告路径")
    p_eval.add_argument("--save-details", default=None,
                        help="保存逐用例评估详情到 JSON 文件 (默认按时间戳自动命名)")
    p_eval.add_argument("--no-save-details", action="store_true",
                        help="不保存逐用例评估详情")
    p_eval.add_argument("--quiet", "-q", action="store_true", help="静默模式，不显示进度条")
    _add_verbose_arg(p_eval)
    _add_agent_arg(p_eval)
    _add_retrieval_args(p_eval)

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_interactive(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
