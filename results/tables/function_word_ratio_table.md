
## Table: Token Type Distribution at Massive Activation Positions

| Model | Function Words (%) | Punctuation (%) | Whitespace (%) | Content Words (%) | **Semantic-Free Total (%)** | Samples |
|-------|-------------------|-----------------|----------------|-------------------|---------------------------|---------|
| GPT-2 | 40.4 | 0.0 | 0.0 | 59.6 | **40.4** | 50 |
| GPT-J-6B | 58.0 | 0.0 | 0.0 | 42.0 | **58.0** | 50 |
| BLOOM-7B1 | 52.0 | 46.0 | 0.0 | 2.0 | **98.0** | 50 |
| Falcon-7B | 0.0 | 0.0 | 100.0 | 0.0 | **100.0** | 50 |
| OPT-6.7B | 0.0 | 96.0 | 0.0 | 4.0 | **96.0** | 50 |
| Mistral-7B | 90.0 | 0.0 | 10.0 | 0.0 | **100.0** | 50 |
| Qwen2.5-7B | 20.0 | 0.0 | 20.0 | 60.0 | **40.0** | 50 |
| LLaMA2-13B | 34.5 | 0.0 | 0.0 | 65.5 | **34.5** | 50 |
| **Average** | - | - | - | - | **70.9** | - |

**Notes:**
- **Function Words**: Articles (the, a), prepositions (in, on, of), conjunctions (and, but), pronouns (it, they), etc.
- **Punctuation**: Commas, periods, parentheses, etc.
- **Whitespace**: Spaces, newlines, tabs.
- **Content Words**: Nouns, verbs, adjectives with semantic meaning.
- **Semantic-Free Total**: Sum of Function Words + Punctuation + Whitespace percentages.

**Key Finding**: On average, **70.9%** of massive activations occur at non-semantic (function word/punctuation/whitespace) positions, supporting the hypothesis that MA serves as a structural marker rather than semantic representation.
