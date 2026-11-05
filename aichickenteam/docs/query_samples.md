# Query plan samples (src/query.py, prompt v11)

Model: Qwen3-4B-Instruct-2507 (bnb 4-bit, `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`), greedy decoding, off-domain 1-shot example + compact JSON prefill.
48 queries: 30 tuning queries (fixed: 1, 2, 3, 5, 51, 211, 258 + random short/mid/long), plus 18 held-out (60 … 1107, incl. 378/1017 "sỏi thận").
Regenerate with `python -m src.query --ids 1,2,3,... --out /tmp/x.jsonl`.

## Throughput (RTX 3050 8GB, shared GPU)
- ~5–6 s/query at batch 8–10 (≈240 generated tokens/query, 80–150 ms per decode step with bnb-4bit). Full 1200 queries ≈ 1.6–2 h.
- Peak VRAM ~4.2 GB (max_batch_tokens 10000) / ~4.7 GB (12000). The shared prompt prefix (system + example, ~700 tokens) is prefilled once and its KV cache reused by every batch.
- Things that helped: compact one-line JSON (prefill `{"terms": [["`), shorter prompt, 1–3 sub-queries instead of 5, prefix KV reuse: 12.7 → ~5 s/query.

## Translation check of the requested terms
| Vietnamese | Output | OK? |
|---|---|---|
| sỏi thận (211, 378, 1017) | kidney stones / renal stone / nephrolithiasis; 肾结石 | yes |
| lao đa kháng (5) | multidrug-resistant tuberculosis (MDR-TB); 耐多药结核病 | yes |
| viêm dạ dày trợt xung huyết (1) | "chronic gastritis with erosive hemorrhage" / erosive hemorrhagic gastritis (ideal: erosive hyperemic/congestive gastritis); 慢性胃炎伴糜烂性出血 | roughly |
| Beta-HCG (3, 1093, 258) | beta-hCG / free beta-hCG; β-hCG | yes |
| hở van 3 lá (948) | tricuspid valve regurgitation; 三尖瓣反流 | yes (after the pitfall list; 1st prompt said mitral) |
| bướu cổ (498) | goiter; 甲状腺肿 | yes (1st prompt: thyroid adenoma) |
| viêm buồng trứng (675) | oophoritis / ovarian inflammation; 卵巢炎 | yes (needed the pitfall list) |

## Known remaining errors (4B model limits)
- 28 "nới răng" (orthodontic separation/expansion) → "tooth extraction".
- 872 "loạn sản sụn xương" (osteochondrodysplasia) → "cartilage degeneration".
- 211 "tán sỏi thận qua da bằng ống soi mềm" → flexible ureteroscopy (ok-ish, the percutaneous part gets lost).
- 1093 "scp" (days after embryo transfer) is sometimes kept as "SCP".
- 722 "rãnh bao quy đầu" (coronal sulcus) → glans/frenulum.
- The model sometimes adds a generic extra sub-question ("Có cần đi khám không?") or leaves out some details; a family member can get swapped ("ba" = father).
- Keywords can contain long phrases rather than atomic terms.
- With an in-domain example (hypertension + diabetes), the model copied those concepts into unrelated plans. Fix: an off-domain example (ankle sprain) plus a leak guard in normalize_plan, which drops example concepts the query does not mention.

## Samples

### id 1

**Q:** Em bị viêm xung huyết 1/3 dưới thực quản, viêm dạ dày trợt xung huyết mạn tính, sẹo loét hành tá tràng. Cho em hỏi các bệnh này của em có nặng và có thể khỏi được không. Em vừa uống xong đợt thuốc được 1 2 ngày thì giờ em lại thấy triệu chứng bị nghẹn ở cổ và hơi buồn nôn. không biết có phải bệnh nặng lên hay do sao.

- **vi:** Viêm xung huyết 1/3 dưới thực quản và viêm dạ dày mạn tính có nặng không và có thể khỏi được không? | Sau khi uống thuốc 1-2 ngày, em lại thấy nghẹt cổ và buồn nôn: có phải bệnh nặng lên không?
- **en:** Distal third esophagitis and chronic gastritis with erosive hemorrhage: are these severe and curable? | After 1-2 days of medication, patient experiences throat obstruction and nausea: could this indicate disease progression?
- **zh:** 下1/3食管炎和慢性胃炎伴糜烂性出血是否严重且可治愈？ | 服药1-2天后出现咽喉阻塞和恶心，是否表示病情加重？
- **pubmed:** `esophagitis distal third severity and prognosis` | `chronic gastritis with erosive hemorrhage treatment outcomes` | `nausea and dysphagia in gastrointestinal ulcer disease` | `gastrointestinal bleeding after anti-ulcer therapy`
- **keywords:** en: esophagitis distal third, chronic gastritis with erosive hemorrhage, duodenal bulb ulceration, pharyngeal obstruction, nausea, severity and prognosis of gastrointestinal disorders; zh: 下1/3食管炎, 慢性胃炎伴糜烂性出血, 十二指肠球部溃疡, 咽喉阻塞, 恶心, 疾病严重性及可治愈性; vi: viêm xung huyết 1/3 dưới thực quản, viêm dạ dày trợt xung huyết mạn tính, sẹo loét hành tá tràng, triệu chứng nghẹt cổ, buồn nôn, bệnh nặng hay khỏi được

### id 2

**Q:** Răng có lỗ chấm đen, chảy máu khi thức dậy là sao?

- **vi:** Răng có lỗ chấm đen và chảy máu khi thức dậy là do sao? | Có cần đi khám nha sĩ không?
- **en:** Black spot and bleeding on tooth upon waking: what is the cause? | Is a dental visit necessary?
- **zh:** 牙齿有黑点且早晨出血是为什么？ | 是否需要看牙医？
- **pubmed:** `black spot on tooth etiology` | `tooth bleeding upon waking` | `dental bleeding morning symptoms`
- **keywords:** en: black spot on tooth, morning bleeding, tooth; zh: 牙齿有黑点, 早晨出血, 牙齿; vi: răng có lỗ chấm đen, chảy máu khi thức dậy, răng

### id 3

**Q:** Kết quả của Beta-HCG là dưới 0.10 U/L thì là như thế nào ạ?

- **vi:** Kết quả Beta-HCG dưới 0.10 U/L có nghĩa là gì? | Kết quả Beta-HCG dưới 0.10 U/L có thể cho biết gì về thai?
- **en:** What does a beta-hCG result below 0.10 mIU/mL mean? | What does a beta-hCG level < 0.10 mIU/mL indicate regarding pregnancy?
- **zh:** β-hCG值低于0.10 mIU/mL表示什么？ | β-hCG值低于0.10 mIU/mL意味着什么？
- **pubmed:** `beta-hCG level < 0.10 mIU/mL pregnancy` | `negative beta-hCG result non-pregnancy` | `beta-hCG test false negative`
- **keywords:** en: beta-hCG, beta-hCG level < 0.10 mIU/mL; zh: β-hCG, β-hCG值<0.10 mIU/mL; vi: beta-hcg, kết quả dưới 0.10 u/l

### id 5

**Q:** Cách phòng tránh nguy cơ lây nhiễm bệnh lao đa kháng là gì?

- **vi:** Cách phòng tránh nguy cơ lây nhiễm bệnh lao đa kháng
- **en:** Methods to prevent transmission risk of MDR-TB
- **zh:** 如何预防耐多药结核病的传播风险
- **pubmed:** `prevention of MDR-TB transmission` | `MDR-TB infection control measures` | `risk factors for MDR-TB transmission`
- **keywords:** en: MDR-TB transmission, prevention, risk of transmission; zh: 耐多药结核病传播, 预防, 传播风险; vi: lây nhiễm lao đa kháng, phòng tránh, nguy cơ lây nhiễm

### id 28

**Q:** Đau 2 bên răng nhai trong quá trình nới răng khi niềng răng có sao không?

- **vi:** Đau răng nhai khi nới răng trong quá trình niềng răng có sao không?
- **en:** Dental pain during mastication when teeth are extracted during orthodontic treatment
- **zh:** 拔牙期间正畸治疗时咀嚼牙痛是否正常？
- **pubmed:** `dental pain during orthodontic tooth extraction` | `mastication pain after tooth extraction in orthodontic patients` | `orthodontic treatment with tooth extraction and pain management`
- **keywords:** en: mastication, dental pain, tooth extraction during orthodontic treatment, orthodontic treatment; zh: 咀嚼, 牙痛, 拔牙期间正畸治疗, 正畸治疗; vi: răng nhai, đau răng, nới răng khi niềng răng, niềng răng

### id 51

**Q:** Cách điều trị đau đầu, tắc mạch máu kèm rối loạn tiền đình?

- **vi:** Cách điều trị đau đầu do tắc mạch máu kèm rối loạn tiền đình? | Rối loạn tiền đình có liên quan đến tắc mạch máu không?
- **en:** Treatment of headache with vascular occlusion and vestibular disorder? | Is vestibular disorder associated with vascular occlusion?
- **zh:** 如何治疗由血管阻塞引起的头痛及前庭功能障碍？ | 前庭功能障碍是否与血管阻塞有关？
- **pubmed:** `headache vascular occlusion treatment` | `vestibular disorder cerebrovascular occlusion` | `cerebral vascular occlusion vestibular symptoms`
- **keywords:** en: headache, vascular occlusion, vestibular disorder, treatment, vestibular dysfunction due to vascular occlusion; zh: 头痛, 血管阻塞, 前庭功能障碍, 治疗, 因血管阻塞导致的前庭功能障碍; vi: đau đầu, tắc mạch máu, rối loạn tiền đình, điều trị, rối loạn tiền đình do tắc mạch

### id 54

**Q:** Đã mổ niệu đạo nhưng có triệu chứng của rối loạn cương dương phải điều trị thế nào?

- **vi:** Sau mổ niệu đạo có triệu chứng rối loạn cương dương thì cần điều trị thế nào? | Rối loạn cương dương sau mổ niệu đạo có phải do phẫu thuật gây ra không?
- **en:** After urethral surgery, how to treat erectile dysfunction? | Is post-urethral surgery erectile dysfunction caused by the surgery?
- **zh:** 尿道手术后出现阳痿应如何治疗？ | 尿道手术后阳痿是否由手术引起？
- **pubmed:** `erectile dysfunction after urethral surgery` | `urethral surgery complications` | `treatment of erectile dysfunction post-surgical`
- **keywords:** en: erectile dysfunction, urethra, urethral surgery, treatment of erectile dysfunction; zh: 阳痿, 尿道, 尿道手术, 阳痿的治疗; vi: rối loạn cương dương, niệu đạo, mổ niệu đạo, điều trị rối loạn cương dương

### id 60

**Q:** Cháu đang là học sinh lớp 10. Gần đây mắt cháu có cảm giác khô liên tục, da mặt cũng xấu đi có thể do tình trạng thiếu hụt vitamin A trong thực đơn hằng ngày. Bác sĩ có thể hướng dẫn cháu cách để cung cấp các loại vitamin cũng như các lượng chất cần thiết để cải thiện tình trạng trên một cách khoa học nhất không.

- **vi:** Có cần bổ sung vitamin A không và bổ sung ở liều bao nhiêu để cải thiện khô mắt và da mặt? | Liều vitamin A phù hợp cho học sinh lớp 10 là bao nhiêu?
- **en:** Is vitamin A supplementation needed, and what is the appropriate daily dose for a 10th-grade student to improve dry eyes and skin? | What is the recommended daily intake of vitamin A for a 10th-grade student?
- **zh:** 是否需要补充维生素A，补充剂量应为多少以改善干眼和面部皮肤状况？ | 10年级学生的维生素A每日推荐摄入量是多少？
- **pubmed:** `vitamin A deficiency dry eyes treatment` | `vitamin A supplementation in adolescents` | `recommended dietary allowance vitamin A for teenagers`
- **keywords:** en: dry eyes, vitamin A deficiency, skin deterioration, vitamin supplementation, recommended daily intake of vitamin A; zh: 干眼, 维生素A缺乏, 皮肤变差, 维生素补充, 所需维生素A摄入量; vi: kém khô mắt, thiếu vitamin A, da mặt xấu, cung cấp vitamin, lượng vitamin cần thiết

### id 121

**Q:** Ngày thứ 2 em xét nghiệm beta được 149.53 đến ngày chậm kinh thứ 9 em có đi siêu âm nhưng chưa thấy túi ối. Bsi chẩn đoán thai chậm phát triển hoặc TNTC, cho em làm xét nghiệm beta được 1285, bác sĩ cho em thuốc tăng nội tiết để bổ sung giúp thai phát triển. Đến ngày 10 sau chậm kinh em xét nghiệm beta 1319. Thực sự em đang rất lo lắng, mong bác sĩ tư vấn tình trạng của em và cách giải quyết.

- **vi:** Tình trạng thai chậm phát triển hoặc TNTC khi siêu âm không thấy túi ối? | Có cần làm xét nghiệm beta-hCG lại không? | Cách điều trị khi beta-hCG tăng nhưng không thấy túi ối trên siêu âm?
- **en:** Ectopic or delayed pregnancy when ultrasound shows no gestational sac? | Is repeat beta-hCG testing necessary? | Treatment approach when beta-hCG increases without visible gestational sac on ultrasound?
- **zh:** 超声未见妊娠囊时，是否为宫外孕或胎发延迟？ | 是否需要重复检测β-hCG？ | β-hCG升高但超声未见妊娠囊时如何治疗？
- **pubmed:** `beta-hCG elevation without gestational sac ultrasound` | `ectopic pregnancy diagnosis criteria` | `delayed gestation with rising beta-hCG` | `hormone supplementation in early pregnancy`
- **keywords:** en: beta-hCG, delayed fetal development, ectopic pregnancy, ultrasound no gestational sac, hormone supplementation; zh: β-hCG, 胎发延迟, 宫外孕, B超未见妊娠囊, 激素补充; vi: beta-hcg, tình trạng thai chậm phát triển, tình trạng thai ngoài tử cung, siêu âm không thấy túi ối, thuốc tăng nội tiết

### id 160

**Q:** Con em năm nay 15 tuổi và đang trong độ tuổi phát triển. Chế độ ăn uống và tập luyện như thế nào để cháu có thể phát triển tốt trong giai đoạn này ạ?

- **vi:** Chế độ ăn uống và tập luyện phù hợp cho trẻ 15 tuổi trong giai đoạn phát triển
- **en:** Appropriate diet and physical exercise for a 15-year-old during adolescence
- **zh:** 15岁青少年青春期发育的合理饮食与运动方案
- **pubmed:** `adolescent growth nutrition guidelines` | `physical activity and puberty development` | `diet and physical exercise in adolescents age 15`
- **keywords:** en: adolescent development, dietary intake, physical exercise, age 15; zh: 青春期发育, 饮食, 体育锻炼, 15岁; vi: phát triển tuổi dậy thì, chế độ ăn uống, tập luyện, tuổi 15

### id 183

**Q:** Cho em hỏi là hôm trước em có đi net thì lỡ uống nhầm nước ngọt của người khác mà chai nước đó đã để được khá lâu và trong nước không còn gas nữa vậy nếu người đó bị hiv thì em có khả năng bị hiv không

- **vi:** Uống nước ngọt của người khác đã để lâu, không còn gas, có bị HIV không? | Nước ngọt không còn gas có thể truyền HIV không?
- **en:** Accidental ingestion of a long-stored, degassed beverage from someone with HIV: risk of infection? | Can degassed beverage transmit HIV?
- **zh:** 误饮他人存放已久的无气饮料，是否会感染HIV？ | 无气饮料能传播HIV吗？
- **pubmed:** `HIV transmission through ingestion of contaminated beverages` | `risk of HIV transmission via oral exposure to non-gaseous liquids` | `HIV transmission routes in accidental exposure scenarios`
- **keywords:** en: accidental ingestion of beverage, depleted carbonation beverage, human immunodeficiency virus, acquisition of HIV through ingestion; zh: 误饮饮料, 无气饮料, HIV, 通过饮用感染HIV; vi: lỡ uống nước ngọt, nước không còn gas, hiv, bị hiv do uống nước

### id 211

**Q:** Tán sỏi thận qua da bằng ống soi mềm áp dụng trong trường hợp nào?

- **vi:** Tán sỏi thận bằng ống soi mềm áp dụng trong trường hợp nào?
- **en:** In which cases is flexible ureteroscopy used for kidney stone treatment?
- **zh:** 软性输尿管镜碎肾结石的适应症是什么？
- **pubmed:** `flexible ureteroscopy indication for kidney stones` | `percutaneous lithotripsy vs flexible ureteroscopy` | `treatment of renal stones with flexible ureteroscopy`
- **keywords:** en: nephrolithotripsy, kidney, flexible ureteroscopy, percutaneous lithotripsy, indications for flexible ureteroscopy; zh: 肾结石, 肾脏, 软性输尿管镜, 经皮肾镜碎石, 适应症; vi: tán sỏi thận, thận, ống soi mềm, tán sỏi qua da, trường hợp áp dụng

### id 258

**Q:** Xin bác sĩ tư vấn giúp em kết quả xét nghiệm sàng lọc trên có bình thường không ạ? Double Test PAPP-A (Cobas) 5.49 Free Beta HCG (Cobas) 84.40 Miễn dịch Hội chứng Down (Trisomy 21) 1:16410 (<1:250) Hội chứng Edward (Trisomy 18) 1:37942 (<1:250) Hội chứng Patau (Trisomy 13) 1:119584.

- **vi:** Kết quả xét nghiệm sàng lọc hội chứng Down, Edward, Patau có bình thường không? | Giá trị PAPP-A và Free Beta HCG có nằm trong ngưỡng bình thường không?
- **en:** Are the results of trisomy 21, 18, and 13 screening tests normal? | Are PAPP-A and free beta-hCG values within normal range?
- **zh:** 唐氏、爱德华氏、帕塔乌综合征筛查结果是否正常？ | PAPP-A 和游离β-hCG 检测值是否在正常范围内？
- **pubmed:** `prenatal screening trisomy 21 PAPP-A` | `free beta-hcg in first trimester screening` | `trisomy 18 and 13 risk assessment`
- **keywords:** en: prenatal screening for trisomy 21, PAPP-A, free beta-hcg, trisomy 18, trisomy 13; zh: 唐氏综合征筛查, PAPP-A, 游离β-hCG, 爱德华氏综合征, 帕塔乌综合征; vi: xét nghiệm sàng lọc hội chứng Down, PAPP-A, Free Beta HCG, hội chứng Edward, hội chứng Patau

### id 276

**Q:** Thai 16 tuần, em bị nứt đầu vú do cơ địa từ trước giờ đã khô sẵn. Em đi tắm và có bị bong lớp vảy sần sùi. Đi khám đầu vú thì bác sĩ dặn bôi kem dưỡng ẩm. Em cũng không sờ nắn hay cậy ti. Mấy hôm nay, thỉnh thoảng lúc ngủ dậy, chợt phát hiện trên áo có vết sữa non rỉ rỉ đã khô. Ti bên phải thỉnh thoảng giật giật nhói nhói. Ra sữa non sớm điều này có ảnh hưởng gì tới sinh non không ạ?

- **vi:** Thai 16 tuần, nứt đầu vú và ra sữa non sớm có ảnh hưởng đến sinh non không? | Có nên lo lắng khi ra sữa non sớm ở thai 16 tuần không?
- **en:** 16-week pregnancy: does early colostrum secretion affect preterm birth? | Should I be concerned about premature lactation at 16 weeks?
- **zh:** 16周妊娠，乳头皲裂和早乳分泌是否会影响早产？ | 16周妊娠出现早乳分泌是否需要担心？
- **pubmed:** `premature lactation in pregnancy` | `nipple fissure during pregnancy` | `colostrum secretion and preterm birth` | `lactation onset in early gestation`
- **keywords:** en: 16-week pregnancy, nipple fissure, colostrum, exfoliation of skin, premature lactation, preterm birth; zh: 16周妊娠, 乳头皲裂, 初乳, 皮肤脱屑, 早乳分泌, 早产; vi: thai 16 tuan, nứt đầu vú, sữa non, bong lớp vảy, ra sữa sớm, sinh non

### id 312

**Q:** Phương pháp khắc phục tác dụng phụ của thuốc điều trị HIV và lao phổi?

- **vi:** Cách khắc phục tác dụng phụ của thuốc điều trị HIV? | Cách khắc phục tác dụng phụ của thuốc điều trị lao?
- **en:** Methods to manage adverse effects of HIV treatment drugs? | Methods to manage adverse effects of tuberculosis treatment drugs?
- **zh:** 如何处理抗HIV药物的不良反应？ | 如何处理抗结核药物的不良反应？
- **pubmed:** `adverse effects management antiretroviral therapy` | `adverse effects of anti-tuberculosis drugs treatment` | `management of drug side effects in HIV and TB co-infection`
- **keywords:** en: adverse effects of antiretroviral therapy, adverse effects of anti-tuberculosis drugs, management of drug side effects; zh: 抗艾滋病药物的不良反应, 抗结核药物的不良反应, 不良反应的处理方法; vi: tác dụng phụ thuốc điều trị HIV, tác dụng phụ thuốc điều trị lao, phương pháp khắc phục

### id 320

**Q:** Cho em hỏi em bị viêm gan B và bị xơ gan độ 3 và độ 4 và sau khi tôi uống thuốc được 2 tháng có đi kiểm tra lại thì giảm xuống độ 2 độ 3. Vậy cho em hỏi liệu gan của em có trở lại bình thường được không?

- **vi:** Viêm gan B và xơ gan độ 3,4 có thể hồi phục được không? | Liệu xơ gan có thể giảm từ độ 3 xuống độ 2 và trở lại bình thường không?
- **en:** Can hepatitis B and stage 3-4 cirrhosis be reversed to normal liver function? | Can cirrhosis regress from stage 3 to stage 2 and return to normal?
- **zh:** 乙型肝炎合并肝硬化3、4期能否恢复？ | 肝硬化能否从3期降至2期并恢复正常？
- **pubmed:** `hepatitis B cirrhosis reversal` | `stage 3 to 4 liver fibrosis regression` | `liver regeneration in advanced cirrhosis` | `reversal of cirrhosis in chronic hepatitis B`
- **keywords:** en: hepatitis B, cirrhosis stage 3 and 4, reversal of cirrhosis from stage 3 to stage 2, liver regeneration to normal; zh: 乙型肝炎, 肝硬化3期和4期, 肝硬化从3期降至2期, 肝脏恢复正常; vi: viêm gan B, xơ gan độ 3 và 4, giảm xơ gan từ độ 3 xuống độ 2, gan hồi phục bình thường

### id 327

**Q:** A-mi-đan sưng là tình trạng của viêm a-mi-đan hay bệnh da liễu?

- **vi:** A-mi-đan sưng là do viêm a-mi-đan hay do bệnh da liễu?
- **en:** Tonsillar swelling: is it due to tonsillitis or a dermatological condition?
- **zh:** 扁桃体肿大是因扁桃体炎还是皮肤疾病引起？
- **pubmed:** `tonsillar swelling etiology` | `tonsillitis vs dermatological swelling` | `inflammatory tonsillar swelling diagnosis`
- **keywords:** en: tonsillar swelling, tonsillitis, dermatological disease; zh: 扁桃体肿大, 扁桃体炎, 皮肤疾病; vi: a-mi-đan sưng, viêm a-mi-đan, bệnh da liễu

### id 349

**Q:** Hồi tháng 6 em có làm siêu âm tuyến giáp và phát hiện khối đồng âm có nhân ở thuỳ phải tuyến giáp, kích thước 7x3mm. Tầm tháng 9 vừa rồi em có làm xét nghiệm chức năng tuyến giáp, các chỉ số T3 FT4 và THS vẫn nằm trong ngưỡng bình thường. Em có nên đi làm xét nghiệm sinh thiết để chắc chắn đây là u lành không? Có hướng điều trị nào cho bệnh của em không?

- **vi:** Khối đồng âm tuyến giáp 7x3mm có cần sinh thiết không? | Có hướng điều trị cho khối u tuyến giáp không?
- **en:** Hypoechogenic thyroid nodule 7x3mm: is biopsy needed? | Treatment options for thyroid nodule
- **zh:** 甲状腺低回声结节7×3mm是否需要活检？ | 甲状腺结节有哪些治疗方案？
- **pubmed:** `hypoechogenic thyroid nodule biopsy indication` | `thyroid nodule management with normal function` | `thyroid nodule treatment guidelines (adult)`
- **keywords:** en: hypoechogenic thyroid nodule, thyroid gland, thyroid biopsy, thyroid function, thyroid treatment; zh: 甲状腺低回声结节, 甲状腺, 甲状腺活检, 甲状腺功能, 甲状腺治疗; vi: khối đồng âm tuyến giáp, tuyến giáp, sinh thiết tuyến giáp, chức năng tuyến giáp, điều trị tuyến giáp

### id 364

**Q:** Dạo này cháu hay đi bơi hôm qua về mắt cháu bị đỏ cháu có sử dụng thuốc nhỏ mắt nhưng cho đến hôm nay mắt cháu sưng đỏ lên bác sĩ có thể cho cháu cách khắc phục không?

- **vi:** Mắt đỏ sau khi đi bơi có cần điều trị không? | Mắt sưng đỏ sau khi dùng thuốc nhỏ mắt có khắc phục được không?
- **en:** Red eye after swimming: is treatment needed? | Can ocular swelling after eye drop use be resolved?
- **zh:** 游泳后眼睛发红是否需要治疗？ | 使用眼药水后眼睛肿红能否缓解？
- **pubmed:** `conjunctivitis after swimming` | `ocular swelling after eye drop use` | `management of red eye in children`
- **keywords:** en: conjunctivitis, ocular swelling, eye drop, management of red eye, ocular inflammation; zh: 结膜炎, 眼肿, 眼药水, 红眼处理, 眼部炎症; vi: mắt đỏ, sưng mắt, thuốc nhỏ mắt, khắc phục mắt đỏ, viêm mắt

### id 378

**Q:** Bác sĩ cho em hỏi nốt vôi hóa thận có thể dẫn tới sỏi thận ạ? Và nang thùy trái có ảnh hưởng nhiều không ạ?

- **vi:** Nốt vôi hóa thận có thể dẫn đến sỏi thận không? | Nang thùy trái có ảnh hưởng đến sỏi thận không?
- **en:** Calcified kidney nodule can lead to kidney stone? | Left renal cyst affects kidney stone formation?
- **zh:** 肾钙化结节是否可能导致肾结石？ | 左肾囊肿是否会影响肾结石形成？
- **pubmed:** `calcified renal nodule kidney stone association` | `renal cyst and risk of nephrolithiasis` | `renal cysts and stone formation pathogenesis`
- **keywords:** en: calcified kidney nodule, kidney stone, left renal cyst, impact on kidney stone formation; zh: 肾钙化结节, 肾结石, 左肾囊肿, 对肾结石形成的影响; vi: nốt vôi hóa thận, sỏi thận, nang thùy trái, tác động đến sỏi thận

### id 379

**Q:** Hôm nay em có làm một số xét nghiệm soi tươi dịch âm đạo, pap,... kết quả hầu như bình thường. Nhưng hôm nay là ngày gần cuối của chu kì kinh nguyệt thì kết quả có chính xác không? Hay phải khám khi sạch kinh 3-5 ngày thì kết quả mới chính xác?

- **vi:** Kết quả xét nghiệm soi tươi và Pap vào gần cuối chu kỳ kinh nguyệt có chính xác không? | Có nên đợi đến khi sạch kinh 3-5 ngày mới xét nghiệm không?
- **en:** Are Pap and wet mount test results accurate near the end of the menstrual cycle? | Should testing be delayed until 3-5 days after menstruation ends?
- **zh:** 月经周期末期做阴道涂片和Pap检查结果是否准确？ | 是否应等到月经干净3-5天后再检查？
- **pubmed:** `wet mount test menstrual cycle timing` | `Pap smear accuracy in late menstrual phase` | `vaginal fluid examination during menstruation`
- **keywords:** en: wet mount of vaginal fluid, Pap smear, menstrual cycle, laboratory test result, timing of test; zh: 阴道分泌物涂片检查, 宫颈细胞学检查, 月经周期, 实验室检查结果, 检查时间; vi: xét nghiệm soi tươi dịch âm đạo, xét nghiệm pap, chu kỳ kinh nguyệt, kết quả xét nghiệm, thời điểm xét nghiệm

### id 461

**Q:** Bị rát mắt lúc đi nắng là do đâu? Khắc phục thế nào?

- **vi:** Tại sao mắt rát khi đi nắng? | Cách khắc phục mắt rát khi đi nắng?
- **en:** Why does the eye hurt in sunlight? | How to relieve eye irritation from sun exposure?
- **zh:** 为什么阳光下眼睛会刺痛？ | 如何缓解阳光引起的结膜刺痛？
- **pubmed:** `solar keratitis etiology` | `sun exposure ocular irritation` | `symptomatic treatment for photokeratitis`
- **keywords:** en: photokeratitis, ocular irritation, symptomatic relief; zh: 紫外线引起的结膜灼伤, 眼睛刺痛, 症状缓解; vi: rát mắt khi đi nắng, mắt rát, khắc phục

### id 498

**Q:** Điều trị bướu cổ cho người bị bệnh cường giáp như nào?

- **vi:** Điều trị bướu cổ ở người bị cường giáp bao gồm những gì? | Các thuốc điều trị cường giáp có ảnh hưởng đến bướu cổ không?
- **en:** What is the treatment of goiter in patients with hyperthyroidism? | Do antithyroid drugs affect goiter in hyperthyroidism?
- **zh:** 甲亢患者甲状腺肿的治疗包括哪些？ | 抗甲状腺药物是否会影响甲亢引起的甲状腺肿？
- **pubmed:** `goiter in hyperthyroidism treatment` | `antithyroid drugs and goiter in Graves' disease` | `management of goiter in hyperthyroidism`
- **keywords:** en: goiter, hyperthyroidism, treatment of goiter, antithyroid drugs; zh: 甲状腺肿, 甲亢, 甲状腺肿的治疗, 抗甲状腺药物; vi: bướu cổ, cường giáp, điều trị bướu cổ, các thuốc điều trị cường giáp

### id 516

**Q:** Em năm nay 22 tuổi sinh bé đướcj hơn 2,5 tuổi em có đặt vòng được hơn 1 năm, mới mấy ngày em tự ý tháo vòng ra vì thấy nó kiểu lồi ra và muốn rơi khỏi tử cung, do công việc bận rộn nên em không có thời gian nghỉ đi khám, không biết tình trạng tự ý tháo vòng của em ra có bị ảnh hưởng gì đến sinh sản sau này không? Hay có bị các vấn đề khác nữa không?

- **vi:** Tự ý tháo vòng tránh thai có ảnh hưởng đến sinh sản sau này không? | Có bị các vấn đề khác sau khi tự ý tháo vòng không?
- **en:** Spontaneous IUD removal: impact on future fertility? | Complications after self-removal of IUD?
- **zh:** 自行取出宫内节育器是否会影响以后生育？ | 自行取出宫内节育器后是否会有其他问题？
- **pubmed:** `spontaneous IUD expulsion fertility outcomes` | `IUD removal complications after self-extraction` | `uterine injury after IUD removal`
- **keywords:** en: intrauterine device, spontaneous IUD removal, fertility impact, complications, uterus; zh: 宫内节育器, 自行取出宫内节育器, 对生育能力的影响, 后续问题, 子宫; vi: vòng tránh thai, tháo vòng tự ý, ảnh hưởng đến sinh sản, vấn đề sau này, tử cung

### id 550

**Q:** Tôi là nam hiện 36 tuổi, bị mỡ máu cao dẫn đến thi thoảng hay đau cả đầu. Nhờ bác sĩ tư vấn giúp về chế độ ăn và thuốc uống nếu cần.

- **vi:** Mỡ máu cao có cần uống thuốc không và chế độ ăn uống nên như thế nào? | Đau đầu do mỡ máu cao có liên quan đến bệnh lý khác không?
- **en:** Is medication needed for hyperlipidemia and what diet is recommended? | Is headache associated with hyperlipemia or another condition?
- **zh:** 高血脂是否需要服药？饮食应如何调整？ | 高血脂引起的头痛是否与其它疾病有关？
- **pubmed:** `hyperlipidemia dietary management` | `hyperlipidemia and headache` | `treatment of hyperlipidemia with statins` | `dietary intervention for hyperlipidemia`
- **keywords:** en: hyperlipidemia, headache, diet, pharmacotherapy, headache due to hyperlipidemia; zh: 高血脂, 头痛, 饮食, 药物治疗, 高血脂引起的头痛; vi: mỡ máu cao, đau đầu, chế độ ăn, thuốc uống, tình trạng đau đầu do mỡ máu cao

### id 572

**Q:** Xin cho em hỏi kinh nguyệt của em rất thưa, thường thì 6 tháng mới có 1 lần. Có cách nào điều chỉnh cho nó ngắn lại không bác sĩ?

- **vi:** Kinh nguyệt thưa 6 tháng 1 lần có cần điều chỉnh không? | Cách điều chỉnh kinh nguyệt để ngắn lại là gì?
- **en:** Infrequent menstruation every 6 months: is it treatable? | How to regulate menstrual cycle to become regular?
- **zh:** 每6个月来一次月经是否可以调节？ | 如何调整月经周期使其变短？
- **pubmed:** `menstrual cycle irregularity treatment` | `luteinizing hormone level in oligomenorrhea` | `hormonal therapy for infrequent menstruation`
- **keywords:** en: menstrual cycle irregularity, menstrual cycle of 6 months, regulate menstrual cycle; zh: 月经稀发, 每6个月一次月经, 调整月经周期; vi: kinh nguyet thua, thua 6 thang 1 lan, chuyen sang kinh nguyet ngan

### id 589

**Q:** Em muốn hỏi là xét nghiệm combo HIV ở tuần thứ 12 và 17 âm tính, giang mai 12 tuần âm tính, bệnh lậu không triệu chứng 17 tuần âm tính nhưng em bị đau họng và trắng lưỡi 2 tháng rồi, nghi bị nấm thì có ảnh hưởng đến xét nghiệm HIV và giang mai không ạ? Em bị viêm gan B nhưng làm định lượng dưới ngưỡng phát hiện thì có cần điều trị gì không ạ?

- **vi:** Xét nghiệm HIV và giang mai âm tính ở tuần 12 và 17 có ảnh hưởng đến triệu chứng đau họng, trắng lưỡi không? | Viêm gan B định lượng dưới ngưỡng phát hiện có cần điều trị không?
- **en:** HIV and syphilis negative at 12 and 17 weeks: does this affect symptoms of sore throat and white tongue? | Hepatitis B viral load below detection limit: is treatment required?
- **zh:** HIV和梅毒在第12、17周检测阴性，是否会影响咽喉痛和舌苔白的症状？ | 乙肝病毒载量低于检测下限是否需要治疗？
- **pubmed:** `oral candidiasis HIV seropositivity` | `syphilis testing false negative in oropharyngeal candidiasis` | `hepatitis B viral load below detection limit management` | `oropharyngeal candidiasis in asymptomatic individuals` | `non-specific sore throat with leukoplakia and hepatitis B`
- **keywords:** en: HIV combination test, syphilis, gonorrhea, hepatitis B, oral candidiasis, pharyngitis, leukoplakia, hepatitis B viral load below detection limit; zh: HIV联合检测, 梅毒, 淋病, 乙型肝炎, 口腔念珠菌病, 咽痛, 舌苔白厚, 乙肝病毒载量低于检测下限; vi: xét nghiệm combo HIV, giang mai, bệnh lậu, viêm gan B, nấm miệng, đau họng, trắng lưỡi, định lượng viêm gan B dưới ngưỡng phát hiện

### id 675

**Q:** Cho em hỏi, tác dụng phụ của vắc-xin 3 trong 1 (sởi + quai bị + rubella) có gây tác dụng phụ gây viêm buồng trứng và vô sinh ở nữ không ạ?

- **vi:** Vắc-xin 3 trong 1 có gây viêm buồng trứng và vô sinh ở nữ không? | Tác dụng phụ của vắc-xin 3 trong 1 là gì?
- **en:** MMR vaccine causing ovarian inflammation and infertility in females? | Adverse effects of MMR vaccine
- **zh:** 麻疹-腮腺炎-风疹疫苗是否会导致女性卵巢炎和不孕？ | 麻疹-腮腺炎-风疹疫苗的不良反应是什么？
- **pubmed:** `MMR vaccine oophoritis` | `MMR vaccine infertility in females` | `adverse effects of MMR vaccine in women`
- **keywords:** en: MMR vaccine, adverse effect, oophoritis, infertility, female; zh: 麻疹-腮腺炎-风疹疫苗, 不良反应, 卵巢炎, 不孕, 女性; vi: vắc-xin 3 trong 1, tác dụng phụ, viêm buồng trứng, vô sinh, phụ nữ

### id 697

**Q:** Cách đây 3 năm, em bị dị ứng mẩn ngứa tôm, cua, cá... uống thuốc tây mới khỏi. Sau đó, em uống thuốc nam thì bớt hẳn. Nhưng hiện tại cơ thể em rất dễ phát nhiệt, lỗ tai hay lùng bùng, da nóng ran do ăn, uống đồ nóng như nước mắm cay, trà gừng hòa tan, vải thiều... Cơ thể em hơi gầy, hay bị đau đầu, choáng mặt và buồn nôn. Em đã từng uống thuốc thiếu máu não, rối loạn tiền đình, viêm xoang... Sắp tới công ty em có đợt chích vắc xin Covid-19 tập trung, em có thể chích trong dịp này không?

- **vi:** Có nên chích vắc xin Covid-19 khi cơ thể dễ phát nhiệt, da nóng ran, đau đầu, buồn nôn không? | Nguyên nhân gây ra cơ thể dễ phát nhiệt, da nóng ran khi ăn đồ nóng là gì? | Có liên quan giữa dị ứng tôm cua cá và hiện tượng dễ phát nhiệt, đau đầu, buồn nôn không?
- **en:** Can I receive COVID-19 vaccine when experiencing feverish reaction, skin flushing, headache, and nausea? | What causes body heat and skin flushing after consuming spicy or hot foods? | Is there a connection between shrimp, crab, fish allergy and symptoms like fever, headache, nausea?
- **zh:** 在身体易发热、皮肤发烫、头痛、恶心时，可以接种新冠疫苗吗？ | 食用辛辣或热食后身体发热、皮肤发烫的原因是什么？ | 虾蟹鱼过敏是否与发热、头痛、恶心有关？
- **pubmed:** `fever and flushing after spicy food intake` | `allergic reaction to seafood and systemic symptoms` | `COVID-19 vaccine safety in patients with autonomic dysfunction or vasomotor symptoms`
- **keywords:** en: allergic rash to shrimp, crab, fish, feverish reaction, ear fullness, skin warmth/flushing, anemia of the brain, vestibular disorder, sinusitis, injection of COVID-19 vaccine; zh: 虾蟹鱼过敏性皮炎, 易发热, 耳闷, 皮肤发烫, 脑贫血, 前庭功能紊乱, 鼻窦炎, 接种新冠疫苗; vi: dị ứng mẩn ngứa tôm cua cá, dễ phát nhiệt, lỗ tai lùng bùng, da nóng ran, thiếu máu não, rối loạn tiền đình, viêm xoang, chích vắc xin Covid-19

### id 722

**Q:** Rãnh bao quy đầu nổi mạch cứng là dấu hiệu của bệnh lý gì?

- **vi:** Rãnh bao quy đầu nổi mạch cứng có phải dấu hiệu của bệnh lý gì không? | Mạch cứng ở rãnh bao quy đầu có liên quan đến bệnh lý nào?
- **en:** Tense vessels in the glans penis: possible underlying pathology? | Hard vessels in the glans penis: associated with which disease?
- **zh:** 阴茎头处出现硬性充血血管是否提示疾病？ | 阴茎头处血管变硬与哪些疾病相关？
- **pubmed:** `venous congestion glans penis` | `tense vessels in glans penis` | `penile vascular abnormalities`
- **keywords:** en: glans penis, venous congestion, tense vessels, penile pathology; zh: 阴茎头, 静脉充血, 硬性血管, 阴茎疾病; vi: rãnh bao quy đầu, nổi mạch, mạch cứng, bệnh lý

### id 789

**Q:** Dạo này, khi vệ sinh vùng hậu môn, em thấy có một mụn, sờ giống như bong bóng cá, không đau và không rát. Có phải em bị lòi trĩ không ạ? Với những biểu hiện trên, em bị trĩ nội hay trĩ ngoại và đang ở cấp độ mấy ạ?

- **vi:** Có phải em bị lòi trĩ không? Có phải là trĩ nội hay trĩ ngoại? | Trĩ nội hay trĩ ngoại và ở cấp độ mấy?
- **en:** Could this be hemorrhoids? Internal or external hemorrhoid? | Internal or external hemorrhoid stage?
- **zh:** 这是否是痔疮？是内痔还是外痔？ | 是内痔还是外痔，属于几期？
- **pubmed:** `painless anal lesion hemorrhoid differential diagnosis` | `external vs internal hemorrhoid clinical presentation` | `hemorrhoid staging clinical criteria`
- **keywords:** en: anal lesion, painless cyst or blister, internal hemorrhoid, external hemorrhoid, hemorrhoid stage; zh: 肛门病变, 水泡样病变, 内痔, 外痔, 痔疮分期; vi: mụn vùng hậu môn, bong bóng cá, trĩ nội, trĩ ngoại, cấp độ trĩ

### id 817

**Q:** Chỉ số creatinin của tôi là 1,54 mg/dL, thận có vấn đề gì không và nên uống thuốc gì để bồi bổ thận?

- **vi:** Chỉ số creatinin 1,54 mg/dL có cho thấy thận có vấn đề không? | Nên uống thuốc gì để bồi bổ thận?
- **en:** Serum creatinine 1.54 mg/dL: does this indicate kidney disease? | What medications are recommended for renal support?
- **zh:** 血清肌酐1.54 mg/dL是否提示肾病？ | 应服用什么药物来补肾？
- **pubmed:** `serum creatinine 1.54 mg/dL kidney dysfunction` | `elevated serum creatinine renal function` | `renal support therapy herbal or conventional`
- **keywords:** en: serum creatinine 1.54 mg/dL, kidney disease, renal support, kidney; zh: 血清肌酐1.54 mg/dL, 肾病, 补肾, 肾脏; vi: creatinine level 1.54 mg/dL, vấn đề thận, bồi bổ thận, thận

### id 820

**Q:** Em mới tiêm phòng dại sau phơi nhiễm mũi cuối cách đây 1 tháng. Vậy nếu bị chuột cắn ở đầu thì phải làm sao ạ?

- **vi:** Sau khi tiêm phòng dại mũi cuối cách đây 1 tháng, nếu bị chuột cắn ở đầu thì phải làm sao? | Có cần tiêm phòng dại bổ sung không?
- **en:** After final rabies vaccine dose one month ago, what if bitten by a mouse on the head? | Is booster rabies vaccination needed after exposure?
- **zh:** 狂犬疫苗最后一针接种一个月后，若被老鼠咬伤头部该怎么办？ | 是否需要加强狂犬病疫苗接种？
- **pubmed:** `rabies post-exposure prophylaxis after final dose` | `rodent bite rabies exposure management` | `rabies vaccination booster after exposure`
- **keywords:** en: rabies vaccination, exposure to rabies, rodent bite, what to do after exposure, last dose; zh: 狂犬疫苗, 狂犬病暴露, 鼠咬, 该怎么办, 最后一针; vi: tiêm phòng dại, phơi nhiễm dại, chuột cắn, phải làm sao, mũi cuối

### id 851

**Q:** Thai em 28 tuần em có đi xét nghiệm nước tiểu. Em bị viêm đường tiết niệu bệnh đó ảnh hưởng tới thai nhi không? Bác sĩ có kê đơn dùng thuốc kháng sinh trong 1 tuần. Uống kháng sinh như thế có ảnh hưởng tới thai nhi không? Ngoài dùng kháng sinh ra em có thể dùng thêm thuốc gì cho khỏi được không?

- **vi:** Viêm đường tiết niệu ở thai 28 tuần có ảnh hưởng đến thai nhi không? | Uống kháng sinh trong 1 tuần có ảnh hưởng đến thai nhi không? | Có thể dùng thêm thuốc gì để điều trị viêm đường tiết niệu không?
- **en:** Urinary tract infection in 28-week pregnancy: impact on fetus? | Antibiotic use during pregnancy: effect on fetus? | Adjunctive medications for urinary tract infection in pregnancy?
- **zh:** 28周妊娠期泌尿系统感染对胎儿有影响吗？ | 妊娠期服用抗生素1周对胎儿有影响吗？ | 可否加用其他药物辅助治疗泌尿系统感染？
- **pubmed:** `urinary tract infection in pregnancy fetal outcome` | `antibiotics in pregnancy fetal safety` | `adjunctive therapy for UTI in pregnancy`
- **keywords:** en: urinary tract infection, 28-week pregnancy, antibiotics, fetal impact, adjunctive medication; zh: 泌尿系统感染, 28周妊娠, 抗生素, 对胎儿的影响, 辅助药物; vi: viêm đường tiết niệu, thai 28 tuần, kháng sinh, ảnh hưởng thai nhi, thuốc bổ trợ

### id 859

**Q:** Trẻ khi khóc hoặc chạy nhảy, bìu thường to một bên nên điều trị thế nào?

- **vi:** Trẻ khóc hoặc chạy nhảy, bìu to một bên có phải viêm bìu không? | Có cần điều trị cho bìu to một bên khi trẻ hoạt động không?
- **en:** Child has enlarged scrotum when crying or running: is it scrotal inflammation? | Is treatment needed for unilateral scrotal swelling during activity?
- **zh:** 孩子哭闹或跳跃时一侧阴囊肿大是否为阴囊炎？ | 孩子活动时一侧阴囊肿大是否需要治疗？
- **pubmed:** `scrotal swelling in children during crying or activity` | `asymmetrical scrotal enlargement in pediatric patients` | `treatment of scrotal swelling in children`
- **keywords:** en: asymmetrical scrotum in crying or activity, scrotal inflammation, scrotal swelling, treatment of scrotal swelling; zh: 一侧阴囊在哭闹或活动时肿大, 阴囊炎, 阴囊肿大, 阴囊肿大的治疗; vi: bìu to một bên khi khóc, viêm bìu, bìu to, điều trị bìu to

### id 872

**Q:** Em muốn hỏi cách điều trị bệnh loạn sản sụn xương.

- **vi:** Cách điều trị bệnh loạn sản sụn xương
- **en:** Treatment of cartilage degeneration disease
- **zh:** 软骨退变疾病的治疗方法
- **pubmed:** `cartilage degeneration treatment` | `osteoarthritis management` | `joint cartilage degeneration therapy`
- **keywords:** en: cartilage degeneration, treatment, osteoarthritis; zh: 软骨退变, 治疗, 骨关节炎; vi: loạn sản sụn xương, điều trị, bệnh loạn sản sụn xương

### id 878

**Q:** Mẹ đang cho con bú có bị giới hạn loại vaccine ngừa Covid-19 hay cân nhắc cai sữa cho bé trước khi tiêm không? Nếu không thì em bé bú sữa mẹ sau khi tiêm có nguy hiểm gì không? Em bé 7 tháng trở lên có được tiêm vaccine Covid-19 không?

- **vi:** Mẹ đang cho con bú có cần cai sữa trước khi tiêm vaccine ngừa Covid-19 không? | Em bé 7 tháng có được tiêm vaccine ngừa Covid-19 không? | Bé bú sữa mẹ sau khi mẹ tiêm vaccine có nguy hiểm không?
- **en:** Is breastfeeding restricted before maternal COVID-19 vaccination? | Can infants 7 months old receive COVID-19 vaccine? | Is there risk to infant when breastfeeding after maternal vaccination?
- **zh:** 哺乳期母亲接种新冠疫苗前是否需要断奶？ | 7个月以上婴儿可以接种新冠疫苗吗？ | 母亲接种疫苗后继续母乳喂养对婴儿有危险吗？
- **pubmed:** `COVID-19 vaccine in breastfeeding mothers` | `maternal vaccination and infant breastfeeding safety` | `vaccination in infants 7 months and above` | `safety of maternal COVID-19 vaccine during lactation`
- **keywords:** en: COVID-19 vaccine, breastfeeding, vaccination in infants, vaccine restriction during breastfeeding, risk to infant after maternal vaccination, infant 7 months; zh: 新冠疫苗, 母乳喂养, 婴幼儿接种, 哺乳期母亲疫苗禁忌, 母乳喂养后婴儿风险, 7个月婴儿; vi: vaccine ngừa covid-19, bú sữa mẹ, tiêm vaccine cho em bé, giới hạn vaccine cho mẹ cho con bú, nguy cơ cho bé bú sữa mẹ sau tiêm, em be 7 thang

### id 885

**Q:** Em có đi khám và bác sĩ nói rằng em bị viêm âm đạo do cầu khuẩn Gram dương (+) và trực khuẩn Gram âm (+). Cho em hỏi trường hợp của em thế này có bị nặng không, và có khả năng lây sang người có quan hệ với em không? Em có nên đưa người có quan hệ với em đi thăm khám hay không, và có nên làm các xét nghiệm về các bệnh lây qua đường tình dục như lậu, HPV, giang mai hay không?

- **vi:** Viêm âm đạo do cầu khuẩn và trực khuẩn có nặng không? | Có lây sang người có quan hệ không? | Có nên đưa người có quan hệ đi khám không? | Có nên làm xét nghiệm lậu, HPV, giang mai không?
- **en:** Vaginal infection due to Gram-positive and Gram-negative bacteria: is it severe? | Can it be transmitted to sexual partners? | Should sexual partners be examined? | Should STI tests (chlamydia, HPV, syphilis) be done?
- **zh:** 由革兰阳性球菌和革兰阴性杆菌引起的阴道炎是否严重？ | 能否传染给性伴侣？ | 是否应让性伴侣去检查？ | 是否应做淋病、HPV、梅毒等性传播疾病检查？
- **pubmed:** `vaginal infection gram positive cocci and gram negative bacilli` | `transmission of vaginal flora to sexual partners` | `screening for STIs in sexual partners of vaginitis` | `chlamydia HPV syphilis co-infection in women with vaginitis`
- **keywords:** en: vaginal infection due to Gram-positive cocci, vaginal infection due to Gram-negative bacilli, sexually transmitted infection, transmission to sexual partners, testing for chlamydia, HPV, syphilis; zh: 阴道感染由革兰阳性球菌引起, 阴道感染由革兰阴性杆菌引起, 性传播疾病, 传染给性伴侣, 检查淋病、HPV、梅毒; vi: viêm âm đạo do cầu khuẩn Gram dương, viêm âm đạo do trực khuẩn Gram âm, bệnh lây qua đường tình dục, lây sang người có quan hệ, xét nghiệm lậu, HPV, giang mai

### id 948

**Q:** Ba tôi năm nay 69 tuổi, có bệnh lý nền: tiểu đường 10 năm nay (đang uống thuốc đều và chỉ số ổn định 5–7 mmol/L), rung tim (mới phát hiện từ cuối năm ngoái), hở van 3 lá (đang theo dõi, chưa kết luận chính xác) vì cần nội soi tim thêm.

- **vi:** Ba tôi 69 tuổi, có tiểu đường 10 năm, chỉ số đường huyết 5–7 mmol/L, rung tim mới phát hiện, hở van 3 lá đang theo dõi, cần nội soi tim thêm – cần tư vấn gì?
- **en:** My father, 69 years old, has 10-year diabetes with blood glucose 5–7 mmol/L, newly diagnosed arrhythmia, tricuspid valve regurgitation under observation, needs cardiac catheterization – what should be advised?
- **zh:** 我父亲69岁，有10年糖尿病，血糖5-7 mmol/L，新发心律失常，三尖瓣反流正在观察，需要做心脏导管检查——需要什么建议？
- **pubmed:** `diabetes mellitus in elderly with arrhythmia` | `tricuspid valve regurgitation in older adults` | `cardiac catheterization indication in valvular heart disease`
- **keywords:** en: comorbidities, diabetes mellitus, arrhythmia, tricuspid valve regurgitation, cardiac catheterization; zh: 合并症, 2型糖尿病, 心律失常, 三尖瓣反流, 心脏导管检查; vi: bệnh lý nền, tiểu đường, rung tim, hở van 3 lá, nội soi tim

### id 1017

**Q:** Tiểu ra máu sau mổ sỏi thận nguyên nhân là gì?

- **vi:** Tiểu ra máu sau mổ sỏi thận có nguyên nhân là gì? | Tiểu máu sau mổ sỏi thận có phải do vết mổ không?
- **en:** What is the cause of hematuria after renal stone surgery? | Is postoperative hematuria due to surgical wound?
- **zh:** 肾结石手术后出现血尿的病因是什么？ | 肾结石手术后血尿是否由手术创面引起？
- **pubmed:** `hematuria after renal stone surgery etiology` | `postoperative hematuria in urological surgery` | `renal stone surgery complications`
- **keywords:** en: hematuria, post-renal stone surgery, etiology; zh: 血尿, 肾结石手术后, 病因; vi: tiểu ra máu, sau mổ sỏi thận, nguyên nhân

### id 1025

**Q:** Con em lúc sinh có xét nghiệm máu gót chân - xét nghiệm sàng lọc rối loạn chuyển hoá bẩm sinh, kết quả có chỉ số Serine (Ser) là 528,15 trong khi ngưỡng cho phép từ 73->350. Bác sỹ có khuyến nghị đi đo định lượng acid amin tuy nhiên do con em mới sinh được 19 ngày nên cho em hỏi chỉ số này có ý nghĩa gì? Có cần thiết phải đi đo định lượng acid amin ngay không?

- **vi:** Chỉ số Serine là 528,15 có ý nghĩa gì trong xét nghiệm sàng lọc rối loạn chuyển hoá bẩm sinh? | Có cần thiết phải đo định lượng acid amin cho bé mới sinh 19 ngày không?
- **en:** What does a serine level of 528.15 mean in newborn metabolic disorder screening? | Is amino acid quantification necessary for a 19-day-old infant with elevated serine?
- **zh:** 新生儿代谢病筛查中丝氨酸水平为528.15有何意义？ | 19天大的新生儿是否需要进行氨基酸定量检测？
- **pubmed:** `serine level in newborn metabolic screening` | `newborn screening inborn errors of metabolism` | `amino acid quantification in neonatal metabolic disorders`
- **keywords:** en: inborn error of metabolism, serine level, newborn screening for metabolic disorders, amino acid quantification, serine reference range; zh: 先天代谢病, 丝氨酸水平, 新生儿代谢病筛查, 氨基酸定量检测, 丝氨酸参考范围; vi: rối loạn chuyển hoá bẩm sinh, chỉ số serine, xét nghiệm sàng lọc rối loạn chuyển hoá, đo định lượng acid amin, ngưỡng cho phép serine

### id 1059

**Q:** Nước tiểu màu đỏ nhạt có nguy hiểm không?

- **vi:** Nước tiểu màu đỏ nhạt có nguy hiểm không?
- **en:** Is pale hematuria dangerous?
- **zh:** 尿液呈淡红色是否危险？
- **pubmed:** `pale hematuria etiology` | `detection of hematuria in adults` | `causes of mild hematuria`
- **keywords:** en: pale hematuria, risk of serious condition; zh: 尿液呈淡红色, 有危险; vi: nước tiểu màu đỏ nhạt, nguy hiểm

### id 1083

**Q:** Em có chu kỳ kinh là 28 ngày. Mấy hôm nay em thử que đều lên hai vạch và rõ dần. Chiều và tối hôm qua em có bị ra máu màu hồng nhạt, sáng nay em thấy ra chút máu nâu hơn nhưng không nhiều. Em không thấy có hiện tượng đau bụng. Siêu âm thì chưa thấy túi ối. Xét nghiệm thì có kết quả như sau. Bác sĩ tư vấn giúp em là liệu có bị thai ngoài tử cung hay dọa sảy không ạ?

- **vi:** Có bị thai ngoài tử cung hay dọa sảy không khi thử que lên hai vạch và siêu âm chưa thấy túi ối? | Giá trị β-hCG có giúp phân biệt thai ngoài tử cung hay dọa sảy không?
- **en:** Is ectopic pregnancy or threatened abortion possible with positive home pregnancy test and no gestational sac on ultrasound? | Can β-hCG levels help differentiate ectopic pregnancy from threatened abortion?
- **zh:** 阳性验孕试纸且B超未见妊娠囊，是否宫外孕或先兆流产？ | β-hCG值能否帮助鉴别宫外孕与先兆流产？
- **pubmed:** `ectopic pregnancy beta-hcg diagnosis` | `threatened abortion ultrasound findings` | `β-hCG level in early pregnancy with absent gestational sac`
- **keywords:** en: ectopic pregnancy, threatened abortion, β-hCG, ultrasound no gestational sac, 28-day menstrual cycle, light pink vaginal bleeding, brown vaginal discharge; zh: 宫外孕, 先兆流产, β-hCG, B超未见妊娠囊, 28天月经周期, 少量鲜红色出血, 褐色分泌物; vi: thai ngoai tu cung, dao sảy, beta-hcg, siêu âm chưa thấy túi ối, chu kỳ kinh 28 ngày, ra máu màu hồng nhạt, ra máu nâu

### id 1093

**Q:** Em chào bác sĩ. Em chuyển phôi ngày 23/9, ngày 2/10 (9 scp) em xét nghiệm beta HCG tại Medlatec kết quả là 224. Hôm nay em xét nghiệm lại (15 scp) kết quả là 281. Bác sĩ cho em hỏi kết quả này có nghi ngờ khả năng thai bị thái hoá không ạ? Vì em không có biểu hiện đau bụng hay ra máu. Vài ngày trước em thấy có đau lâm râm nhưng chỉ 1 lúc thì hết.

- **vi:** Kết quả beta-HCG 224 tại 9 SCP và 281 tại 15 SCP có nghi ngờ thai bị thái hoá không? | Em có cần đi khám thêm không vì không có biểu hiện đau bụng hay ra máu?
- **en:** Beta-hCG level rising from 224 at 9 dpc to 281 at 15 dpc: is spontaneous abortion suspected? | Should I get further evaluation due to absence of abdominal pain or bleeding?
- **zh:** β-hCG值从9天后224增至15天后281，是否怀疑流产？ | 因无腹痛或出血，是否需要进一步检查？
- **pubmed:** `beta-hCG level after embryo transfer pregnancy outcome` | `spontaneous abortion risk in early pregnancy with rising hCG` | `hCG trend in early pregnancy without symptoms`
- **keywords:** en: beta-hCG, spontaneous abortion, embryo transfer, beta-hCG level, abdominal pain; zh: β-hCG, 流产, 胚胎移植, β-hCG值, 腹痛; vi: beta-hcg, thái hoá thai, thời gian chuyển phôi, kết quả beta-hcg, đau bụng

### id 1105

**Q:** Bị chuột rút tay chân làm xét nghiệm máu có ra bệnh hay không?

- **vi:** Chuột rút tay chân có cần làm xét nghiệm máu không? | Xét nghiệm máu có phát hiện ra bệnh không?
- **en:** Muscle cramps in hands and feet: is blood test required? | Can blood test detect underlying disease?
- **zh:** 手足抽筋是否需要做血液检查？ | 血液检查能否发现疾病？
- **pubmed:** `muscle cramp blood test indication` | `muscle cramps as symptom of systemic disease` | `electrolyte imbalance and muscle cramps`
- **keywords:** en: muscle cramp, blood test, disease; zh: 手脚抽筋, 血液检查, 疾病; vi: chuột rút tay chân, xét nghiệm máu, bệnh

### id 1107

**Q:** Viêm bàng quang ở nữ giới có chữa khỏi được không?

- **vi:** Viêm bàng quang ở nữ giới có chữa khỏi được không?
- **en:** Can female cystitis be cured?
- **zh:** 女性膀胱炎能否治愈？
- **pubmed:** `cystitis in females treatment` | `acute cystitis cure rate` | `female urinary tract infection resolution`
- **keywords:** en: cystitis, female, treatment and cure; zh: 膀胱炎, 女性, 治愈; vi: viêm bàng quang, nữ giới, chữa khỏi

### id 1119

**Q:** Mấy ngày gần đây em có triệu chứng hay bị ọe(không phải buồn nôn), ăn không ngon miệng, hay buồn tiểu, có khi vừa đi xong lại buồn tiếp (có lúc buồn nhưng không đi được có lúc lại tiểu dễ dàng, lượng nước tiểu mỗi lần vừa, nước tiểu lúc trong lúc vàng nhạt,không tiểu buốt hay rát, không tiểu đêm). vậy những triệu chứng này liên quan nhiều đến bệnh lý nào. Những triệu chứng này có liên quan đến thủ dâm nhiều không. Dạo gần đây em hay thủ dâm 1 lần/ngày.

- **vi:** Các triệu chứng ọe, ăn không ngon, buồn tiểu có liên quan đến bệnh lý nào? | Triệu chứng buồn tiểu, tiểu dễ dàng, nước tiểu vàng nhạt có liên quan đến thủ dâm không?
- **en:** Symptoms of retching, loss of appetite, urinary urgency: related to which medical condition? | Do frequent self-stimulation cause urinary symptoms like urgency and pale yellow urine?
- **zh:** 干呕、食欲不振、尿频等症状与哪些泌尿系统疾病有关？ | 频繁自慰是否会导致尿频、排尿容易、尿液淡黄等症状？
- **pubmed:** `retching without nausea in adults` | `anorexia and urinary urgency in adults` | `urinary urgency and pale yellow urine etiology` | `self-stimulation and urinary symptoms` | `non-infectious causes of urinary urgency`
- **keywords:** en: retching without nausea, anorexia, urinary urgency, easy urination, pale yellow urine, self-stimulation, urinary system disorder; zh: 干呕, 食欲不振, 尿频, 易排尿, 淡黄色尿液, 自慰, 泌尿系统疾病; vi: ọe (không phải buồn nôn), ăn không ngon miệng, buồn tiểu, tiểu dễ dàng, nước tiểu vàng nhạt, thủ dâm, bệnh lý liên quan

### id 1154

**Q:** Em bầu 20 tuần mà bị ra khí hư nhiều loãng như nước mà không có mùi. Vậy có ảnh hưởng gì không

- **vi:** Bầu 20 tuần ra khí hư nhiều loãng có ảnh hưởng đến thai nhi không? | Có cần đi khám để kiểm tra nguyên nhân không?
- **en:** At 20 weeks pregnancy, odorless watery vaginal discharge: any impact on fetus? | Should I get checked for cause?
- **zh:** 20周妊娠出现无味稀水样白带是否影响胎儿？ | 是否需要检查病因？
- **pubmed:** `leukorrhea in pregnancy` | `vaginal discharge in second trimester` | `vaginal discharge and fetal outcome`
- **keywords:** en: 20-week pregnancy, leukorrhea, odorless, fetal impact; zh: 20周妊娠, 白带增多, 无味, 对胎儿的影响; vi: bầu 20 tuần, ra khí hư nhiều loãng, không có mùi, ảnh hưởng đến thai nhi
