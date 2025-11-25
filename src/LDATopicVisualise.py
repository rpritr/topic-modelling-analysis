from src.TopicVisualise import TopicVisualiseBase

class LDATopicVisualise(TopicVisualiseBase):
    def __init__(self, lda_model, vectorizer, n_topics=None):
        super().__init__(n_topics=n_topics, title_prefix="LDA Topic Modeling")
        self.lda_model = lda_model
        self.vectorizer = vectorizer

    def get_topic_word_frequencies(self):
        feature_names = self.vectorizer.get_feature_names_out()
        topic_freqs = {}

        for topic_idx, topic in enumerate(self.lda_model.components_):
            topic_label = f"Topic {topic_idx + 1}"
            freqs = {feature_names[i]: float(weight) for i, weight in enumerate(topic)}
            topic_freqs[topic_label] = freqs

        return topic_freqs