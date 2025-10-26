from abc import ABC
from transformers import AutoTokenizer, pipeline

class BaseModel(ABC):

    def __init__(self, model_name, from_name, to_name, labels, desc_cat_map):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.from_name = from_name
        self.to_name = to_name
        self.labels = labels
        self.description_to_category = desc_cat_map

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = pipeline(
                task="zero-shot-classification",
                model=self.model_name,
                tokenizer=self.tokenizer
            )

    def predict(self, tasks):
        predictions = []

        self._load_model()

        for task in tasks:
            text = task['data'].get(self.to_name, '').strip()

            if not text:
                predictions.append({'result': [], 'score': 0.0})
                continue

            results = self.model(text, candidate_labels=self.labels)

            choices = [self.description_to_category[label] for label in results['labels'] if
                       label in self.description_to_category]
            scores = [float(s) for s in results['scores']]

            if not choices:
                predictions.append({'result': [], 'score': 0.0})
                continue

            choice = choices[0]
            score = scores[0]

            scores_dict = {
                self.description_to_category[label]: float(score)
                for label, score in zip(results['labels'], results['scores'])
                if label in self.description_to_category
            }

            prediction = {
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': "choices",
                    'value': {"choices": [choice]},
                    'meta': scores_dict
                }],
                'score': score
            }

            predictions.append(prediction)

        return predictions


class ContentTypeModel(BaseModel):

    def __init__(self, model_name='facebook/bart-large-mnli'):
        self.model_name = model_name
        self.from_name = "content_type"
        self.to_name = "body"

        self.description_to_category = {
            'The author expresses frustration, annoyance, or difficulty with a specific situation, product, or process.': 'pain_point',
            'The author highlights something missing, unavailable, or insufficient — often implying a potential improvement or opportunity.': 'unmet_need',
            'The author analyzes, describes, or reflects on trends, behaviors, or problems affecting others or an industry, without personal involvement.': 'observation',
            'The author shares thoughts, insights, or opinions to spark conversation or provide education, without focusing on a specific problem.': 'discussion',
            'The author is experiencing a specific problem, uncertainty, or decision point and seeks guidance, recommendations, or next steps from others.': 'advice_seeking',
            'The author is seeking objective or factual information about a product, process, or situation — not experiencing a problem or paint point.': 'inquiry',
            'The author demonstrates or promotes a product, project, or achievement — either their own or someone else’s.': 'showcase',
            'The post doesn’t fit any other category (spam, irrelevant text, humor, or meta commentary).': 'none'
        }

        self.labels = [
            'The author expresses frustration, annoyance, or difficulty with a specific situation, product, or process.',
            'The author highlights something missing, unavailable, or insufficient — often implying a potential improvement or opportunity.',
            'The author analyzes, describes, or reflects on trends, behaviors, or problems affecting others or an industry, without personal involvement.',
            'The author shares thoughts, insights, or opinions to spark conversation or provide education, without focusing on a specific problem.',
            'The author demonstrates or promotes a product, project, or achievement — either their own or someone else’s.',
            'The post doesn’t fit any other category (spam, irrelevant text, humor, or meta commentary).',
            'The author is experiencing a specific problem, uncertainty, or decision point and seeks guidance, recommendations, or next steps from others.',
        ]

        super().__init__(
            model_name=self.model_name,
            from_name=self.from_name,
            to_name=self.to_name,
            labels=self.labels,
            desc_cat_map=self.description_to_category
        )

    def predict(self, tasks):
        return super().predict(tasks)


class SoftwareRoleModel(BaseModel):

    def __init__(self, model_name='facebook/bart-large-mnli'):
        self.model_name = model_name
        self.from_name = "software_role"
        self.to_name = "body"

        self.description_to_category = {
            'The author is actively searching for a software tool or system to address a problem or improve workflow. ("Looking for", "Is there any software that…")': 'seeking_software',
            'The author is describing or discussing their experience using software, usually in the context of accomplishing something or describing a limitation.': 'using_software',
            'The software itself causes frustration, failure, or is the subject of complaint. ("X keeps crashing", "Y doesn’t sync correctly.")': 'software_is_problem',
            'Software is part of a broader problem or solution, but not the core focus. ("My workflow depends on Google Sheets and email notifications.")': 'software_element',
            'Software is casually mentioned but not related to the problem, goal, or discussion. ("I logged in via Zoom to talk about it.")': 'software_mentioned',
            'The post is about software in general — trends, comparisons, features, or future developments. ("Is Figma killing Sketch?" "Why are people switching to VS Code?")': 'software_discussion',
            'No meaningful software relationship; no software present or relevant.': 'none'
        }

        self.labels = [
            'The author is actively searching for a software tool or system to address a problem or improve workflow. ("Looking for", "Is there any software that…")',
            'The author is describing or discussing their experience using software, usually in the context of accomplishing something or describing a limitation.',
            'The software itself causes frustration, failure, or is the subject of complaint. ("X keeps crashing", "Y doesn’t sync correctly.")',
            'Software is part of a broader problem or solution, but not the core focus. ("My workflow depends on Google Sheets and email notifications.")',
            'Software is casually mentioned but not related to the problem, goal, or discussion. ("I logged in via Zoom to talk about it.")',
            'The post is about software in general — trends, comparisons, features, or future developments. ("Is Figma killing Sketch?" "Why are people switching to VS Code?")',
            'No meaningful software relationship; no software present or relevant.'
        ]

        super().__init__(
            model_name=self.model_name,
            from_name=self.from_name,
            to_name=self.to_name,
            labels=self.labels,
            desc_cat_map=self.description_to_category
        )

    def predict(self, tasks):
        return super().predict(tasks)


class ExternalRoleModel(BaseModel):

    def __init__(self, model_name='facebook/bart-large-mnli'):
        self.model_name = model_name
        self.from_name = "external_role"
        self.to_name = "body"
        self.description_to_category = {
            'The author is seeking help, input, or solutions from an external source — people, organizations, government, or companies.': 'seeking_external',
            'The author is using or leveraging external systems, services, or resources (e.g., "I use Amazon delivery" or "I went to my insurance provider").': 'using_external',
            'The external system or actor is the problem source (e.g., "The landlord refuses to fix it," "My bank keeps freezing my card").': 'external_is_problem',
            'The external entity is mentioned as a contributing part of a broader issue or solution, but not the central one.': 'external_element',
            'The problem arises from broader market, policy, or economic forces (e.g., inflation, housing shortages, labor costs).': 'market_economy_problem',
            "The author’s challenge is personal or situational, not driven by market or software — often individual circumstances or emotions.": 'personal_problem',
            'An external entity is mentioned incidentally — not meaningfully part of the problem or solution.': 'external_mentioned',
            'The post discusses external entities or institutions in general terms — opinion, debate, or analysis ("Why governments fail at innovation").': 'external_discussion',
            'No external entities are involved or relevant.': 'none'
        }
        self.labels = [
            'The author is seeking help, input, or solutions from an external source — people, organizations, government, or companies.',
            'The author is using or leveraging external systems, services, or resources (e.g., "I use Amazon delivery" or "I went to my insurance provider").',
            'The external system or actor is the problem source (e.g., "The landlord refuses to fix it," "My bank keeps freezing my card").',
            'The external entity is mentioned as a contributing part of a broader issue or solution, but not the central one.',
            'The problem arises from broader market, policy, or economic forces (e.g., inflation, housing shortages, labor costs).',
            "The author’s challenge is personal or situational, not driven by market or software — often individual circumstances or emotions.",
            'An external entity is mentioned incidentally — not meaningfully part of the problem or solution.',
            'The post discusses external entities or institutions in general terms — opinion, debate, or analysis ("Why governments fail at innovation").',
            'No external entities are involved or relevant.'
        ]

        super().__init__(
            model_name=self.model_name,
            from_name=self.from_name,
            to_name=self.to_name,
            labels=self.labels,
            desc_cat_map=self.description_to_category
        )

    def predict(self, tasks):
        return super().predict(tasks)
