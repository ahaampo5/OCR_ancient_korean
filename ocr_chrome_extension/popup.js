// Initialize button with user's preferred color
let selectHangeul = document.getElementById("select-hangeul");



// When the button is clicked, inject setPageBackgroundColor into current page
selectHangeul.addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    window.close();
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: setHangeulBox,
    });
  });
  
  
  function setHangeulBox() {
    document.body.style.cursor = "cell";
    document.querySelector('#hangeul-box').style.backgroundColor = 'none';
    document.querySelector('#hangeul-box').setAttribute("data-activate", "true");
    document.querySelector('#overlay').style.display = 'block';
    document.querySelector('#show-box').style.display = 'block';
    
  }