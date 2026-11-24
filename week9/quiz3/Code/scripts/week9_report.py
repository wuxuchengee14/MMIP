"""Build submission reports and figures directly from the final measured results."""
import csv
from collections import Counter
import json
from pathlib import Path

from triage.evaluate import load_benchmark
from triage.paths import ROUTING_DATA
from .week9_llm import OUTPUT as QUIZ2, ROOT, digest

WEEK = ROOT.parent.parent
QUIZ3 = ROOT.parent / 'Output'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, text):
    path.write_text(text.strip() + '\n', encoding='utf-8')


def percent(value):
    return f'{100*value:.1f}%'


def plot_abcd(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = results['methods']
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), layout='constrained')
    positions = range(len(methods))
    axes[0].bar([n-.18 for n in positions], [100*m['schema_ok'] for m in methods.values()], .36, label='Schema', color='#7aaab4')
    axes[0].bar([n+.18 for n in positions], [100*m['intent_ok'] for m in methods.values()], .36, label='Intent', color='#126777')
    axes[0].set(ylabel='Accuracy (%)', ylim=(0, 117), xticks=positions, xticklabels=list(methods))
    axes[0].legend(loc='upper center', ncols=2, frameon=False)
    axes[1].bar(positions, [m['latency_mean_ms'] for m in methods.values()], color='#126777')
    axes[1].set(ylabel='Mean latency (ms, log scale)', yscale='log', xticks=positions, xticklabels=list(methods))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(QUIZ2 / 'benchmark.png', dpi=170)
    plt.close(fig)


def main():
    results = load(QUIZ2 / 'abcd_results.json')
    training = load(QUIZ2 / 'llm_training.json')
    evidence = load(QUIZ3 / 'exp3_results.json')
    prefix = load(QUIZ3 / 'exp3_results_prefix.json')
    cases = load_benchmark()
    methods = results['methods']
    if (list(methods) != list('ABCD') or len(cases) < 10 or any(
        [r['id'] for r in row['records']] != [c['id'] for c in cases] for row in methods.values()
    )):
        raise ValueError('A–D do not share one complete benchmark')
    if results['training_sha256'] != digest(QUIZ2/'llm_training.json') or training['lora_B_change_l2'] <= 0:
        raise ValueError('Missing parameter update evidence')
    if results['benchmark_sha256'] != digest(ROUTING_DATA / 'benchmark.jsonl') or results['generation']['do_sample']:
        raise ValueError('Unexpected benchmark or generation settings')

    with (QUIZ2 / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        table = csv.writer(stream)
        table.writerow(['method','n','schema_ok','route_ok','intent_ok','urgency_ok','exact_ok','mean_ms','p95_ms','estimated_usd_per_1000','api_cost_usd'])
        for name, row in methods.items():
            table.writerow([name, *[row[k] for k in ['n','schema_ok','route_ok','intent_ok','urgency_ok','exact_ok','latency_mean_ms','latency_p95_ms','estimated_usd_per_1000','api_cost_usd']]])
    kind_counts = Counter(case['kind'] for case in cases)
    table = '\n'.join(f"| {key} | {percent(row['schema_ok'])} | {percent(row['route_ok'])} | {percent(row['intent_ok'])} | {percent(row['urgency_ok'])} | {percent(row['exact_ok'])} | {row['latency_mean_ms']:.2f} | {row['latency_p95_ms']:.2f} | {row['estimated_usd_per_1000']:.5f} |" for key, row in methods.items())
    kind_table = '\n'.join(f"| {key} | "+' | '.join(percent(row['by_kind'][kind]['intent_ok']) for kind in ['single','multi','image'])+' |' for key, row in methods.items())
    eval_history = [entry['eval_loss'] for entry in training['history'] if 'eval_loss' in entry]
    write(QUIZ2/'report.md', f"""
# 作業二 四種客服分流方法實測

四組都在同一份 {len(cases)} 筆基準上輸出 `route`、`intent`、`urgency`、`confidence`。題目包含 {kind_counts['single']} 筆單輪、{kind_counts['multi']} 筆多輪及 {kind_counts['image']} 筆圖像案例。圖片題使用原題庫已凍結的 OCR 文字。延遲比較由固定輸入開始計時，圖像上傳與 OCR 的額外耗時由整合端另行處理。

| 方法 | 實作 |
|---|---|
| A | Qwen3-1.7B，只有 system prompt 和三個示例，直接輸出回答。 |
| B | 同一版 Qwen3 加 LoRA SFT，解碼自由。只計算 assistant 完整 JSON 和結束 token 的損失。 |
| C | 同一個 B 的 adapter 和提示。每一步限制可生成 token，前綴樹包含 {results['grammar_candidates']:,} 個合法 JSON 結果，包括 route/intent 的配對及信心值。 |
| D | System One。字元 1–3 gram TF-IDF，兩個 Logistic Regression 分類器分別預測意圖與急迫度，再按規格組出 JSON。 |

B 和 D 使用相同的 {training['train_examples']} 筆分層訓練資料，保留 {training['dev_examples']} 筆驗證資料，核對無完全相同的基準輸入。B 在 {training['gpu']} 上訓練 {training['epochs']:.0f} 個 epoch、{training['steps']} 次更新，耗時 {training['seconds']:.1f} 秒。驗證損失依序為 {', '.join(f'{loss:.5f}' for loss in eval_history)}。LoRA B 權重更新的 L2 差異為 {training['lora_B_change_l2']:.6f}，adapter 雜湊保存在訓練紀錄。

| 方法 | 結構合法 | 部門 | 意圖 | 急迫度 | 三欄皆對 | 平均 ms | P95 ms | 估算美元／千次 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

結構合法指完整字串可解析為 JSON 並通過 schema 與部門、意圖配對檢查。無效結構的內容各欄一律計錯。A–C 使用 BF16、greedy、關閉 thinking、batch 1；D 使用 CPU。預熱、下載及訓練排除在延遲外。API 費用實際為 0，表中成本按實測延遲乘上假設費率 GPU $0.50／小時、CPU $0.04／小時估算，供同批比較。

| 方法 | 單輪意圖 | 多輪意圖 | 圖片意圖 |
|---|---:|---:|---:|
{kind_table}

C 的約束確保這批資料的輸出符合結構，內容意圖仍需單獨評估。C 與 B 使用相同權重，比較的是解碼方式。資料與基準都來自原作業的合成客服情境。即使文字完全匹配已排除，主題與措辭仍可能相近，尤其是 OCR 片段。

![四種方法的結構、意圖與延遲比較](benchmark.png)

[原始輸出](abcd_results.json) · [CSV 比較](comparison.csv) · [實際訓練紀錄](llm_training.json) · [基準題](../Data/benchmark.jsonl) · [模型來源](https://huggingface.co/Qwen/Qwen3-1.7B)
""")

    ux = evidence['E3']
    ood = evidence['E1']
    security = evidence['E2']
    names = list(security['variants'])
    baseline, text_only, full = [security['variants'][name] for name in [names[0], names[-2], names[-1]]]
    original = list(prefix['E2']['variants'].values())[-1]
    selected = next(row for row in evidence['E0']['rows'] if row['tau_fast'] == .5)
    mode_table = '\n'.join(f"| {key} | {row['msgs_per_task']:.2f} | {row['chars_per_task']:.2f} | {percent(row['task_success'])} | {row['wrong_first']} | {row['wrong_final']} |" for key, row in ux['modes'].items())
    m0, m3 = [ood['methods'][m] for m in ['M0','M3']]
    dialogue = ood['dialogue']
    cases_security = sum(security['n_attack_text'].values()) + security['n_attack_image']
    reduction = 1-ux['modes']['confirm']['chars_per_task']/ux['modes']['none']['chars_per_task']
    write(QUIZ3/'report.md', f"""
# 作業三 多輪客服整合與優化

聊天室接收文字及 PNG、JPEG、WebP 圖片。文字、OCR 和已知訂單資料共同支持分流，待補欄位會保留在會話中。右欄顯示路由、意圖、急迫度、信心、耗時與本機工單。已在瀏覽器操作退貨、途中離題與上傳訂單截圖，完成 `request_return` 工單並正確讀出 `A20261104-0087`。

![實際桌面操作](chat_verified.png)

手機響應式版面另有[實際截圖](chat_mobile.png)。整合版使用 System One 快路徑、GRU＋FSM 第二意見和不確定時的澄清。在 {selected['n']} 筆業務題、門檻 0.50 下，覆蓋率為 {percent(selected['coverage'])}，已直接回答的案例意圖正確率為 {percent(selected['precision_on_answered'])}。只使用 D 並一律回答的意圖正確率為 {percent(evidence['E0']['D_only_acc'])}。覆蓋率與正確率需一起解讀。圖片以 OCR 和畫質檢查處理，無文字照片可作為附件交人工查看。工單為本機示範。

## UX 跨話題訂單確認

在 {ux['n_dialogs']} 組模擬對話、{ux['n_tasks']} 個任務中比較四種策略。實際介面預設 `confirm`。完成當前請求後會顯示下一個常見需求的快捷按鈕。

| 策略 | 使用者訊息／任務 | 字數／任務 | 最終任務成功 | 初次錯單 | 最終錯單 |
|---|---:|---:|---:|---:|---:|
{mode_table}

相較每次重問，先確認使輸入字數少 {percent(reduction)}，維持零錯單，訊息數相同。靜默沿用雖然字數最低，卻使 12 筆任務最後掛錯單。可更正策略假定模擬使用者會注意到錯單，因此初次錯單數仍須考慮。

## 聊到無關主題時保留服務進度

使用 {ood['n_ood']} 筆離題與 {ood['n_indomain']} 筆業務題。按開發組選取 M3 的軟澄清門檻 {m3['tau_chosen_on_dev']:.2f}，再在保留一半的測試組回報。測試組離題誤接受從 {percent(m0['test']['ood']['accept'])} 降至 {percent(m3['test']['ood']['accept'])}，同時有 {percent(m3['test']['indomain']['soft'])} 業務題增加一次澄清，硬拒 {percent(m3['test']['indomain']['hard'])}。這是分類策略對照。獨立多輪重播使用聊天室預設門檻，從 {dialogue['baseline_completed']}/{dialogue['n']} 改為 {dialogue['improved_completed']}/{dialogue['n']} 完成原任務。

補件中插入的離題句獨立判斷，保留現有待辦、所缺欄位與訂單確認。回覆會告訴使用者如何接回先前的業務，連續離題可以選真人客服。

## 文字與圖片輸入防禦

使用原有 {cases_security} 筆合成風險案例（{security['n_attack_image']} 筆圖片）、{security['n_benign']} 筆正常案例重跑。先正規化、檢查文字及 OCR 內容，再進入會話狀態和工單政策。上傳端檢查檔案種類及大小。整合測試確認遭攔截的訊息保留原待辦，使用者能繼續完成退貨。

| 本機重測結果 | 既有基準 | 完整防禦 |
|---|---:|---:|
| 未被攔下或導回範圍的比例 | {percent(baseline['asr'])} | {percent(full['asr'])} |
| 正常案例誤擋率 | {percent(baseline['benign_false_positive_all'])} | {percent(full['benign_false_positive_all'])} |
| 易與風險句相似的正常案例誤擋率 | {percent(baseline['benign_false_positive_hard'])} | {percent(full['benign_false_positive_hard'])} |

不掃圖片時，防禦層本身的圖片攔截比例為 {percent(text_only['image']['guard_detected'])}，掃描 OCR 後為 {percent(full['image']['guard_detected'])}。下游分類器在此批圖片也提供保護，因此此比例和整套流程的未攔截比例應分開讀。上游附件把同一批題用於修訂，原始修訂前完整防禦仍有 {percent(original['asr'])} 的案例未被攔截或導回。本次重測用同一組題庫證明既有情境下的改善，樣本範圍限於這組文字、OCR 圖像與模擬會話。

[完整重測資料](exp3_results.json) · [原附件修訂前結果](exp3_results_prefix.json) · [整合測試](verification_tests.txt)
""")
    plot_abcd(results)
    print('Updated Quiz 2 and 3 reports, CSV, and benchmark figure.')


if __name__ == '__main__':
    main()
