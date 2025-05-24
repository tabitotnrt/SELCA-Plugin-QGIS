
document.querySelectorAll('.js-btn').forEach((btn, index) => {
  btn.addEventListener('click', () => {
    const sections = document.querySelectorAll('.js-section');
    sections[index].scrollIntoView({ behavior: 'smooth' });
  });
});
