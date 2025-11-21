from src.TopicVisualise import TopicVisualiseBase


class BertTopicVisualise(TopicVisualiseBase):
    def __init__(self, bertopic_model, n_topics=None):
        super().__init__(n_topics=n_topics, title_prefix="BERTopic")
        self.model = bertopic_model

    def get_topic_word_frequencies(self):
        topic_info = self.model.get_topic_info()

        # izločimo outlier topic (-1)
        topic_ids = topic_info[topic_info["Topic"] >= 0]["Topic"].tolist()

        topic_freqs = {}
        for topic_id in topic_ids:
            words = self.model.get_topic(topic_id)  # list[(word, weight)]
            if not words:
                continue
            freqs = {word: float(weight) for word, weight in words}
            topic_label = f"Topic {topic_id}"
            topic_freqs[topic_label] = freqs

        return topic_freqs