[![Python](https://img.shields.io/badge/python-v3.7-blue)](https://www.python.org/)
[![Linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Docker build: automated](https://img.shields.io/badge/docker%20build-automated-066da5)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://img.shields.io/github/license/pqaidevteam/pqai?style=plastic)](https://github.com/pqaidevteam/pqai/blob/master/LICENSE)

# PQAI Reranker

Service for (re)ranking documents.

Reranking is the process in which a relatively small number of (say a thousand)
documents are ranked in order of their relevance to a query. The small number of
documents are typically obtained by a coarse but fast searching technique from a
large corpus. Reranking techniques are more accurate in terms of relevance
judgement but not scalable for millions of documents.

<br>
For more detailed information, please refer to [PQAI_Wiki](https://github.com/pqaidevteam/pqai/wiki/pqai-reranker).

## Routes

| Method | Endpoint  | Comments                                                           |
| ------ | --------- | ------------------------------------------------------------------ |
| `GET`  | `/rerank` | Sort given documents w.r.t. their relevance to a given query       |

## How to run?

### From command line

1. Clone this repository
1. Download required [assets](https://s3.amazonaws.com/pqai.s3/public/assets-pqai-reranker.zip) and extract them to `/assets` directory
1. Create a `.env` file using `/env` template and set environment variable values
1. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
1. Make sure the [encoder service](https://github.com/pqaidevteam/pqai-encoder) is running and properly configured in `.env` file
1. Run the service: `python3 main.py`

### As docker container

1. Clone this repository
1. Create a `.env` file using `/env` template and set environment variable values
1. Give execution permission to the deployment script: `chmod +x deploy.sh`
1. Run deployment script: `bash deploy.sh`

## License

The project is open-source under the MIT license.

## Contribute

We welcome contributions.

To make a contribution, please follow these steps:

1. Fork this repository.
2. Create a new branch with a descriptive name
3. Make the changes you want and add new tests, if needed
4. Make sure all tests are passing
5. Commit your changes
6. Submit a pull request

## Support

Please create an issue if you need help.
