const lightbox = document.querySelector(".lightbox");
const lightboxImage = lightbox.querySelector("img");
const lightboxTitle = document.querySelector("#lightbox-title");

document.querySelectorAll(".result-card").forEach((card) => {
  card.addEventListener("click", () => {
    lightboxImage.src = card.dataset.image;
    lightboxImage.alt = card.querySelector("img").alt;
    lightboxTitle.textContent = card.dataset.title;
    lightbox.showModal();
  });
});

lightbox.querySelector(".lightbox__close").addEventListener("click", () => lightbox.close());
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.close();
});

document.querySelector("#year").textContent = new Date().getFullYear();
