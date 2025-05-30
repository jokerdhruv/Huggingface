from llama_cpp import Llama

# Load the Llama model
model = Llama(
    model_path="llama-2-7b.Q2_K.gguf",  # Path to GGUF file
    n_ctx=2048, 
    n_gpu_layers=0  # Use 0 for CPU
)

# Define the input prompt
input_text = """Summarize the following text: The Indian independence movement was a series of historic events in South Asia with the ultimate aim of ending British colonial rule. It lasted until 1947, when the Indian Independence Act 1947 was passed.

The first nationalistic movement for Indian independence emerged in the Province of Bengal. It later took root in the newly formed Indian National Congress with prominent moderate leaders seeking the right to appear for Indian Civil Service examinations in British India, as well as more economic rights for natives. The first half of the 20th century saw a more radical approach towards self-rule.

The stages of the independence struggle in the 1920s were characterised by the leadership of Mahatma Gandhi and Congress's adoption of Gandhi's policy of non-violence and civil disobedience. Some of the leading followers of Gandhi's ideology were Jawaharlal Nehru, Vallabhbhai Patel, Abdul Ghaffar Khan, Maulana Azad, and others. Intellectuals such as Rabindranath Tagore, Subramania Bharati, and Bankim Chandra Chattopadhyay spread patriotic awareness. Female leaders like Sarojini Naidu, Vijaya Lakshmi Pandit, Pritilata Waddedar, and Kasturba Gandhi promoted the emancipation of Indian women and their participation in the freedom struggle.

Few leaders followed a more violent approach, which became especially popular after the Rowlatt Act, which permitted indefinite detention. The Act sparked protests across India, especially in the Punjab Province, where they were violently suppressed in the Jallianwala Bagh massacre.

The Indian independence movement was in constant ideological evolution. Essentially anti-colonial, it was supplemented by visions of independent, economic development with a secular, democratic, republican, and civil-libertarian political structure. After the 1930s, the movement took on a strong socialist orientation. It culminated in the Indian Independence Act 1947, which ended Crown suzerainty and partitioned British India into the Dominion of India and the Dominion of Pakistan. On 26 January 1950, the Constitution of India established the Republic of India. Pakistan adopted its first constitution in 1956.[1] In 1971, East Pakistan declared its own independence as Bangladesh.""
"""




# Run the model inference
response = model(input_text, max_tokens=100)  # Generate up to 100 tokens
print("Summary:", response["choices"][0]["text"].strip())