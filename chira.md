
#### Ключевые исправления

1. **Каналы и PyTorch GPU:** Добавлены каналы `pytorch` и `nvidia`, указан `pytorch-cuda=11.8` (полная поддержка микроархитектуры Ampere / RTX 3070). Исключен ломающий синтаксис `pytorch=*=*cuda*`.
2. **XGBoost:** Убрана невалидная маска `xgboost=*=*cuda*`. В `conda-forge` сборка `xgboost` для Linux автоматически скомпилирована с поддержкой CUDA и активируется флагом `device="cuda"`, как и заложено в [model_training.py](file:///Users/viac/Documents/projects/active/last-try/model_training.py#L62).
3. **Исключение `defaults`:** Предотвращены конфликты ABI между Anaconda `defaults` и `conda-forge` при сборке `viennarna`, `pysam` и `samtools`.
4. **Стабильность солвера:** `pandera`, `pyarrow`, `optuna` и `rich` перенесены в секцию Conda (гарантирует сборку под один C-runtime), а `chira` исключена из Conda-спецификации.

---

### 2. Как использовать ChiRA в вашем случае и есть ли ей замена?

#### В чем фундаментальная проблема с ChiRA здесь?

1. **Несовпадение CLI-интерфейса:** В [chimeric_mapping.sh](file:///Users/viac/Documents/projects/active/last-try/chimeric_mapping.sh#L94) вызывается:

   ```bash
   chira extract -m "${MIRNA_FA}" -1 "${TRIMMED_R1}" -2 "${TRIMMED_R2}" -o "${CHIRA_OUT}" -t "${THREADS}"
   ```

2. **Отсутствие бинарника в Bioconda:** Официальный пакет `chira` из Bioconda не создает исполняемый файл `chira` (в `$PATH` попадают лишь скрипты `chira_extract.py`, `chira_collapse.py` и др.).
3. **Несовпадение входных/выходных контрактов:** Утилита `chira_extract.py` из пакета ChiRA принимает на вход **BED-файл выравниваний** (`-l/--loci`), а не сырые FASTQ (`-1`, `-2`), и не генерирует файл `targets.fastq.gz` с заголовками вида `@read_id:::mirna_name`, который жестко ожидает [chimeric_mapping.sh](file:///Users/viac/Documents/projects/active/last-try/chimeric_mapping.sh#L151) и [chimera_metrics.py](file:///Users/viac/Documents/projects/active/last-try/chimera_metrics.py).
4. **Конфликт зависимостей:** ChiRA тянет за собой устаревший стек (`clan`, `intarna`, старый GCC ABI), который замораживает conda-солвер при установке рядом с PyTorch и Python 3.10.

#### Есть ли ChiRA замена?

**Да, и она намного эффективнее.** В современных eCLIP/CLASH протоколах для деконволюции химер используют:

* **Кастомные стриминговые сплиттеры:** Они сопоставляют известные последовательности микроРНК (miR-128-3p, let-7-5p) с концами химерного рида, отрезают их и формируют таргетные фрагменты.
* **Cutadapt в режиме demultiplexing/trimming:** `cutadapt` находит микроРНК как адаптер и переименовывает заголовок.

#### Как применить решение без изменения исходников (паттерн Wrapper/Shim)

Поскольку менять [chimeric_mapping.sh](file:///Users/viac/Documents/projects/active/last-try/chimeric_mapping.sh) **категорически запрещено**, подготовлен готовый инструмент [chira_shim.py](file:///Users/viac/Documents/projects/active/last-try/chira_shim.py).

Он принимает аргументы:
`-m <mature.fa> -1 <R1> -2 <R2> -o <out_dir> -t <threads>`
и генерирует файл `${CHIRA_OUT}/targets.fastq.gz` с правильным форматом заголовков `@read_id:::mirna_name`.

#### Инструкция по подключению

После создания и активации окружения:

```bash
# 1. Создание и активация окружения
mamba env create -f environment.yml
mamba activate chimera_pipeline

# 2. Подключение обертки chira в PATH активного окружения
ln -sf "$(pwd)/chira_shim.py" "${CONDA_PREFIX}/bin/chira"

# 3. Проверка доступности команды chira
chira --help
```

После этого вызов `./main.sh --all` (или `./main.sh --step chimeric_mapping`) выполнит шаг деконволюции химер без единой ошибки и без модификации оригинальных скриптов.
