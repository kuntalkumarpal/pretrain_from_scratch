from transformers import RobertaConfig
from transformers import RobertaForMaskedLM

# Setting newer config
config = RobertaConfig(
                        hidden_size= 512, #intermediate layer
                        max_position_embeddings= 1026, #Sequence length
                        num_attention_heads= 8,
                        num_hidden_layers= 6,
                        vocab_size= 50265,
                        type_vocab_size=1,
                        intermediate_size = 2048,
                        layer_norm_eps = 1e-05,
                    )
model = RobertaForMaskedLM(config=config)


print("Number of Parameters of the model:",model.num_parameters())
# => 84 million parameters


#Saving the model architecture
model.save_pretrained("/scratch/kkpal/NEW_VAREC/models/base_model")
