import requests
import time
from PyPDF2 import PdfReader

def get_ids(num_page):
    headers = {
        'Authorization': 'Guest',
    }

    params = {
        'count': '10',
        'page': num_page,
    }

    while True:
        response = requests.get('https://consultas.anvisa.gov.br/api/consulta/bulario', params=params, headers=headers)
        response_json = response.json()

        if 'content' in response_json:
            return response_json['content']
        elif 'error' in response_json and response_json['error'] == 'Session is closed!':
            print("erro de sessão fechada, tentando novamente")
            time.sleep(5)
        else:
            return None
#-----------------------------------------------------------
def quantity():
    headers = {
        'Authorization': 'Guest',
    }

    params = {
        'count': '10',
        'page': '1',
    }

    response = requests.get('https://consultas.anvisa.gov.br/api/consulta/bulario', params=params, headers=headers)
    response_json = response.json()
    return response_json
#----------------------------------------------------------
def download_pdf_from_anvisa_api(id, save_path):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            time.sleep(2**attempt)  # Adiciona um atraso entre as tentativas
            url_bula = f"https://consultas.anvisa.gov.br/api/consulta/medicamentos/arquivo/bula/parecer/{id}/?Authorization="
            response = requests.get(url_bula)
            if response.status_code == 200:
                with open(save_path, 'wb') as file:
                    file.write(response.content)
                print("Arquivo salvo com sucesso:", save_path)
                return True  # Retorna True se o download for bem-sucedido
            else:
                print(f"Tentativa {attempt+1}: Erro ao baixar o arquivo. Código de status:", response.status_code)
                print(response.text)
                
        except Exception as e:
            print(f"Tentativa {attempt+1}: Erro na função:", e)
            time.sleep(3)  # Adiciona um atraso entre as tentativas

    # Se todas as tentativas falharem, retorna False
    return False
#----------------------------------------------------------

totalelementos = quantity()
print(totalelementos)
totalPages = totalelementos.get("totalPages")

while totalPages == None:    
    totalelementos = quantity()
    totalPages = totalelementos.get("totalPages")
    #aguarda 10 segundos para tentar novamente
    print("aguardando 10 segundos para tentar novamente")
    time.sleep(10)



print(f"o total de paginas é: {totalPages}")

for page in range(7, totalPages):

    # ID do medicamento e caminho onde deseja salvar o PDF
    bulas = get_ids(page+1)
    while bulas == None:
        bulas = get_ids(page+1)
        time.sleep(10)
    
    print(f"A PAGINA ATUAL É: {page}")
    for b in bulas:
        idProduto = b['idProduto']
        nomeProduto = b['nomeProduto']
        idBulaPaciente = b['idBulaPacienteProtegido']
        nome_arquivo = str(idProduto)+".pdf"
        pagina = (page+1)

        # Tentativa de baixar o PDF
        if not download_pdf_from_anvisa_api(idBulaPaciente, "./bulas/"+str(idProduto)+".pdf"):
            print(f"Falha ao baixar o PDF para {nomeProduto} (ID: {idProduto})")
        
            
        

    time.sleep(1)


