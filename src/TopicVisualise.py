import matplotlib.pyplot as plt
from wordcloud import WordCloud


class TopicVisualise:
    
    def __init__(self, lda_model, vectorizer, n_topics):
        self.lda_model = lda_model
        self.vectorizer = vectorizer
        self.n_topics = n_topics

    def visualize_topics_wordcloud(self):
        """Generira wordcloud vizualizacijo za vsako temo"""
        
        print("\n" + "="*80)
        print("GENERIRANJE WORDCLOUD VIZUALIZACIJ...")
        print("="*80)
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Nastavi figuro za vse wordcloud-e
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Topic Modeling - WordCloud Vizualizacija', fontsize=16, fontweight='bold')
        
        # Flatten axes za lažje iteriranje
        axes = axes.flatten()
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            if topic_idx >= self.n_topics:
                break
                
            # Ustvari slovar besed in njihovih uteži
            word_weights = {}
            for i, weight in enumerate(topic):
                word_weights[feature_names[i]] = weight
            
            # Generiraj wordcloud
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate_from_frequencies(word_weights)
            
            # Prikaži wordcloud
            axes[topic_idx].imshow(wordcloud, interpolation='bilinear')
            axes[topic_idx].set_title(f'Tema {topic_idx + 1}', fontsize=14, fontweight='bold')
            axes[topic_idx].axis('off')
        
        # Skrij dodatne subplot-e, če jih je več kot tem
        for idx in range(self.n_topics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Shrani sliko
        output_file = 'topic_wordclouds.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Wordcloud vizualizacija shranjena kot: {output_file}")
        plt.close()