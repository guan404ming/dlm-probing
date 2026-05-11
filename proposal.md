DLM + SAE 用於 Functional Correctness 研究重點
核心命題
從 probe-based correctness prediction 延伸到 SAE-based,把單一 scalar 訊號升級為可分解、可干預的 feature dictionary。
與 prior work (我自己的 probe 論文) 的關係
	∙	Probe 論文建立了「DLM 內部表徵能預測 correctness」這個 prior
	∙	SAE 工作是延伸,不是修正 — 同一個訊號從黑盒 scalar 變成結構化 feature
	∙	Framing:research program,不是打補丁
Probe → SAE 的本質差異
	∙	Probe 是監督式、單一目標、read-only
	∙	SAE 是非監督的通用 feature dictionary、read-write (能 steering)
	∙	同一組 SAE 同時支援 early stop / decoding order / error diagnosis / steering
三層故事架構
	1.	預測 — SAE feature probe 在 correctness prediction 上 match 或微贏 raw hidden state probe (sanity check + baseline)
	2.	診斷 — Fail case 的 feature signature clustering,做 error taxonomy:off-by-one / hallucinated API / missing return 等可命名 error mode
	3.	修正 — 在 denoising 中途 suppress error feature,把 fail case 救回 pass
Code DLM 的獨特紅利
	∙	雙向 context:feature 可能編碼「等待 return」、「預期後面有 try/except」這類 look-ahead 訊號
	∙	AR + code SAE 結構上做不到,純 DLM bonus
早期 sanity check (做之前必做)
拿現成 DLM-Scope SAE,在你 probe 用的 hidden state 上 encode,重訓 probe。
	∙	贏 → 故事全部成立
	∙	輸 → DLM-Scope SAE 沒抓到 code reasoning,需要訓 code-specialized SAE
Early stop 作為附加章節 (optional)
	∙	現有 DLM early stop (Prophet / EDIT / Just on Time) 全都用輸出端或 gradient 訊號
	∙	SAE 是內部表徵訊號,能抓 false confidence (logit 穩了但 feature 還在飄)
	∙	跟你的 probe 故事可以共用同一套 infrastructure
潛在反駁與回應
	∙	「Probe 已經 work,SAE 多此一舉」→ 必須有結構性贏的維度,主推 read-write (steering 修正) 和 multi-dimensional diagnosis
	∙	「SAE encode overhead」→ 報 wall-clock 不只 step 數
	∙	「Error taxonomy 定性無依據」→ 需要量化驗證機制 (下個 session 要討論)
待解問題 (下個 session 可以接著聊)
	∙	Error taxonomy 怎麼避免被嫌主觀
	∙	兩篇論文 narrative 怎麼引、怎麼定位
	∙	DLM-Scope SAE 不夠用時,自己訓 code SAE 的 cost / 規劃