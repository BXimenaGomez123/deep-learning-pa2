default:
  @just --choose

download-dataset:
    if [ ! -d "data/MOT17" ]; then \
      curl -L -o data/MOT17.zip https://motchallenge.net/data/MOT17.zip; \
      unzip -q data/MOT17.zip -d data; \
      rm data/MOT17.zip; \
    fi
    @printf "Train: %s entries\n" $(ls "data/MOT17/train" | wc -l)
    @printf "Test: %s entries\n" $(ls "data/MOT17/test" | wc -l)

download-dataset-labels:
    if [ ! -d "data/MOT17Labels" ]; then \
      curl -L -o data/MOT17Labels.zip https://motchallenge.net/data/MOT17Labels.zip; \
      unzip -q data/MOT17Labels.zip -d data/MOT17Labels; \
      rm data/MOT17Labels.zip; \
    fi
    @printf "Train: %s entries\n" $(ls "data/MOT17Labels/train" | wc -l)
    @printf "Test: %s entries\n" $(ls "data/MOT17Labels/test" | wc -l)

test:
    uv run pytest

typecheck:
    uvx ty check

all_checks: typecheck test
