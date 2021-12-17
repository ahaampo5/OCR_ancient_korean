const SERVERL_URL ='http://ec2-3-34-44-46.ap-northeast-2.compute.amazonaws.com:6006/recognize';

fetch(chrome.runtime.getURL('/template.html')).then(r => r.text()).then(html => {
    document.body.insertAdjacentHTML('beforeend', html);
  });

let selectBoxX = -1
let selectBoxY = -1


document.body.addEventListener('click', e=>{
    //하단에 박스 닫는 버튼 클릭시
    if (e.target.id =='hangeul-close'){
        let showBox = document.querySelector('#show-box');
        let hangeulBox = document.querySelector('#hangeul-box');
        let hangeulOutput = document.querySelector('#hangeul-output');
        let hangeulImage = document.querySelector('#hangeul-image');

        //관련된 UI들 초기화
        showBox.style.display = 'none';
        hangeulBox.style.width='0px';
        hangeulBox.style.height='0px';
        hangeulBox.style.top="-1px";
        hangeulBox.style.left="-1px";
        hangeulOutput.value = '';
        hangeulImage.src = 'https://via.placeholder.com/800x200?text=image';        
    }
});


document.body.addEventListener('mousedown', e => {
    var isActivated = document.querySelector('#hangeul-box').getAttribute("data-activate");
    let hangeulBox = document.querySelector('#hangeul-box');
    hangeulBox.style.display='block';

    if(isActivated=="true"){
        let x = e.clientX;
        let y = e.clientY;
        //Select Box의 시작점을 현재 마우스 클릭 지점으로 등록
        selectBoxX = x;
        selectBoxY = y;
        
        //hangeul Box이 위치와 사이즈를 현재 지점에서 초기화
        hangeulBox.style.top = y+'px';
        hangeulBox.style.left = x+'px';
        hangeulBox.style.width='0px';
        hangeulBox.style.height='0px';
    }
});


//캡쳐가 준비된 상태에서 (마우스 클릭이 된 상태) 드래그시 박스 사이즈 업데이트
document.body.addEventListener('mousemove', e => {
    try{
        var hangeulBox = document.querySelector('#hangeul-box');
        var isActivated = hangeulBox.getAttribute("data-activate");
    }catch(e){
        return;
    }
    
    //팝업에서 Start 버튼을 클릭하고, select 박스의 값이 초기값이 아닌 상태인 경우 시작
    if(isActivated=="true" && (selectBoxX != -1 && selectBoxY != -1)){
        let x = e.clientX;
        let y = e.clientY;

        //Select 박스(hangeul-box)의 가로 세로를 마우스 이동에 맞게 변경
        width = x-selectBoxX;
        height = y-selectBoxY;
        
        hangeulBox.style.width = width+'px';
        hangeulBox.style.height = height+'px';
        
    }
});

// 마우스 드래그가 끝난 시점 (드랍)
document.body.addEventListener('mouseup', e => {
    let hangeulBox = document.querySelector('#hangeul-box');
    let isActivated = hangeulBox.getAttribute("data-activate");
    
    //만약 팝업의 start 버튼을 클릭한 후의, 그냥 취소
    if(isActivated=="false"){
        return ;
    }

    // 다음 이벤트가 ??
    hangeulBox.setAttribute("data-activate", "false");
    
    
    
    let x = parseInt(selectBoxX);
    let y = parseInt(selectBoxY);
    let w = parseInt(hangeulBox.style.width);
    let h = parseInt(hangeulBox.style.height);
    
    //캡쳐 과정이 끝났으므로, hangeul-box 관련된 내용 초기화
    selectBoxX = -1;
    selectBoxY = -1;
    
    
    hangeulBox.style.display='none';
    hangeulBox.style.width='0px';
    hangeulBox.style.height='0px';
    hangeulBox.style.top="-1px";
    hangeulBox.style.left="-1px";

    //Overaly 화면 안보이게 초기화
    document.querySelector('#overlay').style.display='none';
    //마우스 Cursor도 원래 커서로 초기화
    document.body.style.cursor = "default";


    //200ms 정도의 시간차를 두고 서버로 현재 캡쳐된 이미지를 전송
    //시간차를 안두면, 박스와 오버레이 화면이 같이 넘어갈 수 있음
    setTimeout(function(){
        chrome.runtime.sendMessage({text:"hello"}, function(response) {
            var img=new Image();
            img.crossOrigin='anonymous';
            img.onload=start;
            img.src=response;
            
            function start(){
                //화면 비율에 따라 원래 설정한 좌표 및 길이와 캡쳐본에서의 좌표와 길이가 다를 수가 있어서, 그에 대응하는 비율을 곱해줌
                ratio = img.width/window.innerWidth;
                
                
                var croppedURL=cropPlusExport(img,x*ratio,y*ratio,w*ratio,h*ratio);
                var cropImg=new Image();
                cropImg.src=croppedURL;
                document.querySelector('#hangeul-image').src = croppedURL;
                fetch(SERVERL_URL, {
                    method: 'POST',
                    body: JSON.stringify({"image":croppedURL}), // data can be `string` or {object}!
                    headers:{
                        'Content-Type': 'application/json'
                    }
                }).then(res => res.json())
                .then(response => {    
                    document.querySelector('#hangeul-output').value = response['result'];
                });
            }
    
        });
    },200);
   
    
});

//전체 스크린샷을 crop하는 함수
function cropPlusExport(img,cropX,cropY,cropWidth,cropHeight){
    
    
    var canvas1=document.createElement('canvas');
    var ctx1=canvas1.getContext('2d');
    canvas1.width=cropWidth;
    canvas1.height=cropHeight;
    
    ctx1.drawImage(img,cropX,cropY,cropWidth,cropHeight,0,0,cropWidth,cropHeight);
    
    return(canvas1.toDataURL());
  }


