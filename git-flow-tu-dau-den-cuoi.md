# Git Flow – Từ đầu đến cuối

Git Flow là một mô hình phân nhánh Git giúp team quản lý việc phát triển tính năng, chuẩn bị release và xử lý lỗi production một cách rõ ràng.

## 1. Các nhánh chính trong Git Flow

### `main`
- Chứa code đang chạy trên production.
- Mỗi commit trên `main` nên tương ứng với một bản phát hành ổn định.
- Thường được gắn tag như `v1.0.0`, `v1.0.1`.

### `develop`
- Là nhánh tích hợp cho quá trình phát triển hằng ngày.
- Các tính năng mới sẽ được merge vào đây trước khi release.

### `feature/*`
- Dùng để phát triển từng tính năng riêng lẻ.
- Được tách ra từ `develop`.
- Khi hoàn thành, tạo Pull Request để merge lại vào `develop`.

### `release/*`
- Dùng để chuẩn bị phát hành.
- Được tách ra từ `develop` khi team muốn chốt bản release.
- Thường dùng để fix bug nhỏ, cập nhật version, kiểm tra cuối trước khi lên production.
- Sau đó merge vào cả `main` và `develop`.

### `hotfix/*`
- Dùng để sửa lỗi khẩn cấp trên production.
- Được tách ra từ `main`.
- Sau khi sửa xong, merge vào cả `main` và `develop`.

---

## 2. Quy trình Git Flow từ đầu đến cuối

## Bước 1: Khởi tạo project

```bash
# Tạo thư mục project
mkdir my-project
cd my-project

# Khởi tạo Git
git init

# Tạo file đầu tiên
echo "# My Project" > README.md

# Commit đầu tiên
git add .
git commit -m "feat: initial commit"

# Đổi tên branch thành main (nếu cần)
git branch -M main
```

Hiện tại có:
- `main`

---

## Bước 2: Tạo nhánh `develop`

```bash
# Tạo nhánh develop từ main
git checkout -b develop

# Khai báo remote
git remote add origin git@github.com:user/repo.git

# Push các nhánh lên remote
git push -u origin main
git push -u origin develop
```

Hiện tại có:
- `main`
- `develop`

Lúc này hai nhánh đang giống nhau.

---

## Bước 3: Phát triển tính năng đầu tiên

```bash
# Luôn cập nhật develop trước khi tách feature
git checkout develop
git pull origin develop

# Tạo nhánh feature từ develop
git checkout -b feature/add-login

# Code tính năng login
echo "function login() {}" > login.js

# Commit
git add .
git commit -m "feat: add login function"

# Push branch feature lên remote
git push -u origin feature/add-login
```

Hiện tại có:
- `main`
- `develop`
- `feature/add-login`

---

## Bước 4: Tạo Pull Request và merge vào `develop`

Trên GitHub/GitLab:
- Tạo Pull Request từ `feature/add-login` → `develop`
- Review code
- Approve và merge

Sau khi merge trên web, ở máy local chỉ cần cập nhật lại `develop`:

```bash
git checkout develop
git pull origin develop
```

Sau đó có thể xóa branch feature:

```bash
git branch -d feature/add-login
git push origin --delete feature/add-login
```

Hiện tại có:
- `main` (chưa đổi)
- `develop` (đã có login)

> Lưu ý: Nếu team không dùng PR/MR thì có thể merge local bằng `git merge --no-ff`, nhưng trong thực tế đa số team sẽ merge trên GitHub/GitLab.

---

## Bước 5: Dev khác làm tính năng thứ hai

```bash
# Cập nhật develop mới nhất
git checkout develop
git pull origin develop

# Tạo branch feature mới
git checkout -b feature/add-register

# Code tính năng register
echo "function register() {}" > register.js

# Commit và push
git add .
git commit -m "feat: add register function"
git push -u origin feature/add-register
```

Sau đó:
- Tạo Pull Request từ `feature/add-register` → `develop`
- Merge trên GitHub/GitLab
- Cập nhật lại local:

```bash
git checkout develop
git pull origin develop
```

Hiện tại có:
- `main` (chưa đổi)
- `develop` (đã có login + register)

---

## Bước 6: Chuẩn bị release lên production

Khi team muốn phát hành phiên bản mới, tạo nhánh release từ `develop`.

```bash
git checkout develop
git pull origin develop
git checkout -b release/v1.0.0

# Ví dụ cập nhật version
echo "version: 1.0.0" > version.txt
git add .
git commit -m "chore: bump version to 1.0.0"

# Push release branch
git push -u origin release/v1.0.0
```

Trong giai đoạn này có thể:
- fix bug nhỏ
- update version
- kiểm thử kỹ trước khi release

Hiện tại có:
- `main`
- `develop`
- `release/v1.0.0`

---

## Bước 7: Release lên production (`main`)

Trên GitHub/GitLab:
- Tạo Pull Request từ `release/v1.0.0` → `main`
- Merge sau khi kiểm thử xong

Sau đó cập nhật local và tạo tag:

```bash
git checkout main
git pull origin main

git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin main
git push origin --tags
```

Tiếp theo cần merge ngược release vào `develop` để tránh mất các commit chỉnh sửa trong quá trình release:

Trên GitHub/GitLab:
- Tạo Pull Request từ `release/v1.0.0` → `develop`
- Merge

Sau đó cập nhật local:

```bash
git checkout develop
git pull origin develop
```

Cuối cùng xóa nhánh release:

```bash
git branch -d release/v1.0.0
git push origin --delete release/v1.0.0
```

Hiện tại có:
- `main` (`v1.0.0`)
- `develop` (đã đồng bộ nội dung của release)

---

## Bước 8: Phát hiện bug khẩn cấp trên production

Nếu production đang có lỗi khẩn cấp, phải tách `hotfix` từ `main` vì `main` là code đang chạy thực tế.

```bash
git checkout main
git pull origin main
git checkout -b hotfix/fix-login-error

# Sửa bug
echo "function login() { /* fixed */ }" > login.js

# Commit
git add .
git commit -m "fix: resolve login timeout error"

# Push hotfix branch
git push -u origin hotfix/fix-login-error
```

Sau đó:
- Tạo Pull Request từ `hotfix/fix-login-error` → `main`
- Merge trên GitHub/GitLab

Cập nhật local, tạo tag hotfix và push tag:

```bash
git checkout main
git pull origin main

git tag -a v1.0.1 -m "Hotfix 1.0.1"
git push origin main
git push origin --tags
```

Tiếp theo phải merge hotfix vào `develop` để nhánh phát triển cũng có bản sửa lỗi này:

- Tạo Pull Request từ `hotfix/fix-login-error` → `develop`
- Merge trên GitHub/GitLab

Cập nhật local:

```bash
git checkout develop
git pull origin develop
```

Xóa nhánh hotfix:

```bash
git branch -d hotfix/fix-login-error
git push origin --delete hotfix/fix-login-error
```

Hiện tại có:
- `main` (`v1.0.1`)
- `develop` (`v1.0.1`)

---

## Bước 9: Tiếp tục phát triển tính năng mới

Sau khi hotfix hoặc release xong, team quay lại quy trình bình thường:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/add-profile
```

Sau đó tiếp tục:
- code
- commit
- push
- tạo PR vào `develop`
- merge

Chu kỳ tiếp tục lặp lại.

---

## 3. Tóm tắt quy trình

1. Khởi tạo project với nhánh `main`
2. Tạo `develop` từ `main`
3. Tạo `feature/*` từ `develop` → merge lại vào `develop`
4. Tạo `release/*` từ `develop` → merge vào `main` và `develop`
5. Tạo `hotfix/*` từ `main` → merge vào `main` và `develop`
6. Tiếp tục lặp lại chu trình

---

## 4. Sơ đồ timeline

```text
Thời gian →

main:     ●────────────────────●(v1.0.0)──●(v1.0.1)────→
          │                    │          │
          │                    │       hotfix
          │                 release      │
          │                    ╲         │
develop:  ●────●────●────●─────●─────────●────●────→
               │    │                        │
            feature feature                feature
            (login)(register)             (profile)
```

---

## 5. Lưu ý thực tế

### Nên dùng Pull Request thay vì merge local
Nếu team đang dùng GitHub/GitLab thì nên:
- push branch lên remote
- tạo Pull Request / Merge Request
- review code trên web
- merge trên web
- local chỉ cần `git pull`

Cách này giúp lịch sử rõ ràng hơn và phù hợp quy trình teamwork.

### Có thể dùng `--no-ff` nếu merge local
Nếu merge local, nên dùng:

```bash
git merge --no-ff feature/add-login
git merge --no-ff release/v1.0.0
git merge --no-ff hotfix/fix-login-error
```

Việc này giúp giữ lại dấu vết của từng nhánh trong lịch sử commit.

### Git Flow không phải lúc nào cũng là lựa chọn tốt nhất
Git Flow phù hợp khi:
- team có quy trình release rõ ràng
- có staging/UAT
- có nhiều người cùng làm việc
- cần tách biệt mạnh giữa dev và production

Nếu team nhỏ và deploy liên tục, nhiều nơi sẽ dùng mô hình đơn giản hơn như:
- GitHub Flow
- trunk-based development

---

## 6. Kết luận

Git Flow giúp tổ chức quá trình phát triển phần mềm một cách bài bản:
- phát triển tính năng trên `feature/*`
- tích hợp ở `develop`
- chốt bản phát hành qua `release/*`
- sửa lỗi production qua `hotfix/*`
- giữ `main` luôn ổn định để deploy

Nếu team có nhiều môi trường và quy trình release rõ ràng, Git Flow là một lựa chọn rất dễ quản lý.
