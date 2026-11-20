# Evaluation results

37 cases in `eval/cases.json` (English and Arabic). Run on 2026-11-20.

## Retrieval: is an expected project in the top 3?

No model calls. Keyword = the search before vectors.

| Question | Keyword | Hybrid |
| --- | --- | --- |
| Trump villas in Oman | ✅ | ✅ |
| Tell me about DG1 | ✅ | ✅ |
| Apartments with Aston Martin interiors | ✅ | ✅ |
| The tower designed with Pagani in Dubai | ✅ | ✅ |
| What do you have in Spain? | ✅ | ✅ |
| Any DarGlobal project in Doha? | ✅ | ✅ |
| A luxury residence in London | ✅ | ✅ |
| Homes with Missoni interiors | ✅ | ✅ |
| A family resort with kids entertainment in Oman | ✅ | ✅ |
| A community built around padel courts | ✅ | ✅ |
| Villas with Fendi Casa interiors | ✅ | ✅ |
| Is there a Trump tower in Saudi Arabia? | ✅ | ✅ |
| Golf course villas in Dubai | ✅ | ✅ |
| Marriott branded homes | ✅ | ✅ |
| Mouawad interiors in Riyadh | ✅ | ✅ |
| seaside home with a private beach | ✅ | ✅ |
| مشاريع دار جلوبال في إسبانيا | ❌ | ✅ |
| أخبرني عن برج ترامب جدة | ❌ | ✅ |
| شقق فاخرة في الدوحة | ❌ | ✅ |
| فلل ترامب في عمان | ❌ | ✅ |
| I want to buy a Trump Cliff Villa, my budget is 3M AED and I want to move within 3 months | ✅ | ✅ |
| **Project hit@3** | **17/21 (81%)** | **21/21 (100%)** |

## Retrieval: is the right company page (about, FAQ, visas, offices) in the top 2?

| Question | Hybrid |
| --- | --- |
| Who is the CEO of DarGlobal and is it a listed company? | ✅ |
| Can I get a golden visa if I buy in Dubai? | ✅ |
| Where is your office in London? | ✅ |
| What documents do I need to buy a DarGlobal property? | ✅ |
| ما هي خطط الدفع المتاحة؟ | ✅ |
| Can I buy a fraction of a property with tokens? | ✅ |
| **Company page hit@2** | **6/6** |
