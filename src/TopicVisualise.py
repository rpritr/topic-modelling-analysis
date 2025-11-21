from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class TopicVisualiseBase(ABC):
    def __init__(self, n_topics=None, title_prefix="Topic Modeling"):
        """
        n_topics    : maksimalno število tem za prikaz (če None -> vse)
        title_prefix: prefix za naslov figure
        """
        self.n_topics = n_topics
        self.title_prefix = title_prefix

    @abstractmethod
    def get_topic_word_frequencies(self):
        """
        Vrne:
            Ordered dict / navaden dict oblike:
            {
                topic_label_1: { "word1": weight1, "word2": weight2, ... },
                topic_label_2: {...},
                ...
            }
        topic_label je lahko int ali str (npr. 0, 1, 2 ali "Topic 0").
        """
        pass

    def visualize_topics_wordcloud(self, output_file="topic_wordclouds.png"):
        """Generira wordcloude za vse teme, na osnovi podatkov iz get_topic_word_frequencies()."""
        print("\n" + "=" * 80)
        print(f"GENERATING WORDCLOUDS ({self.__class__.__name__})...")
        print("=" * 80)

        topic_freqs = self.get_topic_word_frequencies()
        if not topic_freqs:
            print("⚠ Ni podatkov o topicih.")
            return

        # omejimo število tem, če je potrebno
        topic_items = list(topic_freqs.items())
        if self.n_topics is not None:
            topic_items = topic_items[:self.n_topics]

        n = len(topic_items)

        # layout
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
        fig.suptitle(self.title_prefix + " - WordClouds", fontsize=16, fontweight="bold")

        # če je samo en row/col, axes ni nujno array
        if not isinstance(axes, (list, tuple)):
            try:
                axes = axes.flatten()
            except Exception:
                axes = [axes]
        else:
            axes = [ax for sub in axes for ax in (sub if isinstance(sub, (list, tuple)) else [sub])]

        for idx, (topic_label, freqs) in enumerate(topic_items):
            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="viridis",
                relative_scaling=0.5,
                min_font_size=10,
            ).generate_from_frequencies(freqs)

            axes[idx].imshow(wc, interpolation="bilinear")
            axes[idx].set_title(str(topic_label), fontsize=14, fontweight="bold")
            axes[idx].axis("off")

        # skrij odvečne osi
        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\n✓ Wordcloud saved as: {output_file}")
        plt.close()