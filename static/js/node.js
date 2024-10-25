document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('loginModal');
    const openModalButton = document.getElementById('openModal');
    const closeButton = document.querySelector('.close-button');
    const loginForm = document.getElementById('loginForm');

    // Open modal
    openModalButton.addEventListener('click', () => {
        modal.style.display = 'block';
    });

    // Close modal
    closeButton.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    // Close modal when clicking outside
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });

    // Handle form submission
    loginForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;

        // Replace this with your actual login logic
        console.log(Username: ${username}, Password: ${password});

        // Close the modal after submission
        modal.style.display = 'none';
    });

    // Handle social login buttons (add your own logic)
    document.querySelector('.google-button').addEventListener('click', () => {
        console.log('Google sign up clicked');
        // Add Google sign-in logic here
    });

    document.querySelector('.linkedin-button').addEventListener('click', () => {
        console.log('LinkedIn sign up clicked');
        // Add LinkedIn sign-in logic here
    });
});