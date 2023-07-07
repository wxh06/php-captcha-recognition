# luogu-captcha-bypass

## Dependencies

```bash
# Python dependencies for generating data & training
pip install -r requirements.txt

# PHP dependencies for generating
composer install

# (optional) Node.js dependencies for API server
pnpm install
```

## Generating data

```bash
# Install dependencies
pip install -r requirements.txt
composer install

python generate.py
```

## Training

```bash
# Install dependencies
pip install -r requirements.txt

python train.py
```

## API server

```bash
# Install dependencies & build
pnpm install
pnpm run build

pnpm run start
```
