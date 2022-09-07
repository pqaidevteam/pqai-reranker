[![Python](https://img.shields.io/badge/python-v3.7-blue)](https://www.python.org/)
[![Linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Docker build: automated](https://img.shields.io/badge/docker%20build-automated-066da5)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://img.shields.io/github/license/pqaidevteam/pqai?style=plastic)](https://github.com/pqaidevteam/pqai/blob/master/LICENSE)

_Note: This repository is under active development and not ready for production yet._

# PQAI Reranker

Service for (re)ranking documents.

Reranking is the process in which a relatively small number of (say a thousand)
documents are ranked in order of their relevance to a query. The small number of
documents are typically obtained by a coarse but fast searching technique from a
large corpus. Reranking techniques are more accurate in terms of relevance
judgement but not scalable for millions of documents.

## Routes

| Method | Endpoint  | Comments                                                           |
| ------ | --------- | ------------------------------------------------------------------ |
| `GET`  | `/rerank` | Sort given documents w.r.t. their relevance to a given query       |

## Deploy

Production

```
docker-compose up
```

Development
```
docker-compose -f docker-compose.dev.yml up
```

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
