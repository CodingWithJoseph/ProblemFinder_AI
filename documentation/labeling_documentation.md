# Issues With Labeling
## General Notes or Improvements
- Structure post to included 5-10 examples: 1/3 positives, 1/3 negatives, 1/3 edge cases
- Include Decision Rules, Edge Case Rules Or Descriptions
- Observation — Consistency across near-duplicate posts:
- Multiple posts across different subreddits describe identical problems with slight rewording.
- Model output remains consistent (e.g., same label set and reasoning).
- Indicates schema stability and semantic generalization.
- Action: add duplicate detection during preprocessing to avoid class skew in training/evaluation.
- If a post shares advice, workflow, or solution but clearly references a real problem or frustration that others face, label is_problem = 1 (software_solvable = 0, external = 0).
- Only label is_problem = 0 if the post has no problem context at all (pure showcase, news, or opinion).
- There are post that describe a problem and present a solution, however some should be labeled as 0 0 0 and others 1 1 0. How do you make this distinction without confusing the model? (ZCU106 Simulink vs My Top 3 Tools for Ditching 90% of My Typing) Solution may require multiple models to look for different aspects of post and ensemble these together

## Is Problem
### Edge Case to Remember
- **Randomly got a package in the mail (1 0 0 → 0 0 0):** The post only describes a past problem that is already resolved or expresses relief or gratitude, treat it as NOT an active problem (is_problem = 0).
- **Entry-level job market rant + advice (1 1 0 → 1 0 0):** The post discusses a real, widespread problem (difficulty finding entry-level developer jobs) but the author is sharing their own advice and workflow rather than seeking help. It still reflects a genuine market pain point, so it should be labeled as a problem (`is_problem = 1`), but not software-solvable (`is_software_solvable = 0`) or external (`is_external = 0`). This aligns with the project goal of capturing real-world frustrations that signal potential product or service opportunities.
- **Looking for beginner architecture software (0 0 0 → 1 0 0):** The post reflects a genuine user need — wanting affordable software to start learning architecture due to space limitations for drawing. It’s a *problem* (`is_problem = 1`) because the user is seeking a solution to a constraint, but it’s *not software-solvable* (`is_software_solvable = 0`) since they’re asking for existing tools, not describing a flaw or gap that could be solved by building new software. It’s also *not external* (`is_external = 0`).

### Notes:

## Is Software Solvable
### Edge Cases to Remember
- **Samsung Odyssey+ blurry image (1 0 0 → 1 1 0):** The post describes a hardware symptom (blurriness) but attributes it to potential software or driver changes. Since the cause or fix is likely software-related, it should be labeled as software-solvable (is_software_solvable = 1) even though the issue presents as hardware.
- **Data event stream (1 1 0 → 1 0 0):** The post describes a real problem — the user feels lost and is seeking guidance on where to start. However, there is no specific technical failure or solvable software issue; it’s a learning and support problem, not a software or infrastructure malfunction. Therefore, mark as a problem (is_problem = 1) but not software-solvable or external.
- **Looking to make a waterfall chart (1 1 0 → 1 0 0):** The post describes a problem — the user is seeking guidance on how to create a specific type of plot. However, the issue is not a software bug or malfunction; it’s an information or know-how problem. They’re asking how to implement something, not reporting that existing software is broken. Therefore, label it as a problem (is_problem = 1) but not software-solvable (is_software_solvable = 0) since the solution is learning guidance, not a code or configuration fix.
- **ZCU106 Simulink / Embedded Coder setup (1 1 0 → 1 0 0):** The post describes a clear problem — the user is seeking guidance on how to configure and deploy a model-based design flow for a specific evaluation board. However, there is no malfunction or software failure; it’s an information-seeking / setup issue, not a bug or defect. The problem cannot be directly “solved” by changing code or applying a patch — it requires knowledge or experience.
- **Vevor Smart1 cutter software rant (1 1 0 → 1 0 0):** The post expresses frustration with bundled software and restrictive licensing but does not request help or describe a solvable malfunction. The user has already found an alternative (Easy Cut Studio) and is sharing information with others. This is an opinion/experience post rather than a current software problem needing resolution.

### Notes

## Requires External Component
### Edge Cases to Remember
- **Horse blanket LoRa sensor prototype (1 1 0 → 1 1 1):** The post describes a clear problem — designing a temperature + humidity monitoring system — and seeks technical guidance. The solution involves software and hardware integration (software-solvable = 1), but it also depends on external components, manufacturing, and physical deployment (external = 1).
- **Choosing a first 3D printer (1 0 0 → 1 0 1):** The post reflects a real problem — the user is confused and searching for a 3D printer that meets their needs — but they’re explicitly looking for an existing product to solve it. Under a solution-centric framework, this means the user’s need already maps to available market options rather than an unsolved gap. So the correct label is *problem = 1*, *software-solvable = 0*, *external = 1*.
- **Rivian Gear Guard unreliable; seeking dashcam alternatives (1 0 0 → 1 0 1):** Clear problem (Gear Guard failing with USB-C storage; missed events), but the practical fix is either a manufacturer firmware update or buying an existing dashcam. That means it’s *not* software-solvable by a third party (`is_software_solvable = 0`), and the user is explicitly seeking a market solution (`is_external = 1`). So the correct label is *problem = 1*, *software-solvable = 0*, *external = 1*.
- **GoXLR Mini malfunction and search for alternatives (1 0 0 → 1 0 1):** The post describes a clear problem — the GoXLR Mini crashes, losing volume control during streams — but the user has already accepted it as failing and is now actively searching for an existing replacement. This makes it *problem = 1* (genuine frustration), *software-solvable = 0* (the issue is hardware-related and proprietary), and *external = 1* (the user is looking for available market products to fix it). The corrected label is **(1 0 1)**.
- **Which job should I pursue? (1 0 0 → 1 0 1):** The post shows a real problem — uncertainty about whether to continue as an SDET or pivot careers — but the user is explicitly looking for guidance and existing career paths, not a new product or software solution. The solution lies in external options like education or job opportunities, not in creating new tools. So the correct label is *problem = 1*, *software-solvable = 0*, *external = 1*.
- **Elderly relative struggling after Samsung software update (1 0 0 → 1 0 1):** The post expresses a real problem — the latest Samsung update made the phone harder for an elderly user to navigate — but the user is looking for an existing fix or rollback option, not proposing or needing new software. The solution lies in available system settings or third-party tools, not in developing new technology. So the correct label is *problem = 1*, *software-solvable = 0*, *external = 1*.



