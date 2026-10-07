# Explain this project out loud

Practice these in your own words. Short answers beat perfect ones.

## 30-second pitch
"I built a model that flags order lines likely to lose money. A simple rule, discount over 20%, already catches most losses, so the real question was whether a model could also find the losers hiding in the 1-20% discount range. On orders the model never saw, a random forest caught 83% of losing lines against 70% for the rule, and flagged 96.5% of the loss dollars against 89%. It's a modest gain, and I tested it honestly."

## Numbers to know
- 9,994 order lines; 18.7% lose money
- Split by date: 60% train, 20% validation, 20% final exam (newest)
- Rule (discount > 20%): 70% of losing lines caught, 97% of its flags right
- Random forest: PR-AUC 0.953; 83% caught; 87.7% of flags right (target was 90%)
- Dollars on the exam period: $36.6K lost; rule flags $32.7K; model flags $35.3K

## Questions you will get

**Why split by date, not randomly?**
New orders come after old ones. A random split lets the model peek at the future, so the score looks better than it would be in real use.

**What is leakage? Where could it have happened here?**
Using information the model wouldn't have at decision time. `profit`, `cost` and `margin_pct` are calculated from the answer, so I left them out. The automatic checker missed them; I caught them by thinking about how each column is made.

**Why not just report accuracy?**
81% of lines don't lose money, so "never flag anything" already scores 81%. Accuracy hides the thing we care about.

**What is PR-AUC, in plain words?**
A score for how well the model ranks the money-losing lines above the safe ones, built for cases where the thing you want is the minority. 1.0 is perfect; guessing scores about the loss share (0.19).

**Why a random forest and not XGBoost, which scored a bit higher?**
Before training I wrote a rule: if two models are within 0.005 PR-AUC, pick the simpler one. XGBoost was ahead by 0.0048, so the forest won. The three models were basically tied; I said so in the report.

**What is the final exam, and why touch it once?**
The newest 20% of orders, locked away until the end. If you keep checking it and adjusting, it stops being a fair test. I scored it once and reported the number even though one part missed the target.

**What went wrong or disappointed you?**
Flags were right 87.7% of the time on the exam, below the 90% I wanted. The cut-off I picked on validation didn't carry over perfectly to newer data. I didn't re-tune it, because that would have meant peeking.

**What mistake did you catch along the way?**
My first dollar-impact numbers were wrong: I matched profit to the wrong rows and it showed only 16% of losses flagged. A check that the joined data matched the labels failed, so I fixed the join by using order keys and added the check to the script.

**Is the model worth using?**
For this store, it adds about $2.6K of flagged loss over the rule in six months. That is a small gain. The honest use is as a review aid, and the rule is a strong baseline that is hard to beat by much.

**How would you keep it working over time?**
Check monthly that loss rates and discount patterns look like training. Retrain every six months. Pick a new cut-off on the newest data. Retrain sooner if recall falls below 70% or precision below 80%.

**What would you do differently with more time?**
Use real insurance data, add cost-based flagging (weigh by dollars, not just count), and test on another store or a longer period.

**Who wrote the code?**
Say it plainly: "I built it with Claude Code. I set the question, approved each stage, checked the results, and I can explain every decision." Then be ready to prove it with the questions above.

## Weak spots to admit before they ask
- One store's data, one dataset
- The three models are within noise of each other
- The gain over the rule is real but small
- Flags are a review aid; they don't save money by themselves
