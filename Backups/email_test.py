from pushover import Client

client = Client("u3mra51amezcopes8nx5csnkm2xn11", api_token="acqnqxckdz3hhaqkn1gebpj2vq9hcb")

client.send_message("Hello!", title="Hello")
