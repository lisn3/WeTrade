pragma experimental ABIEncoderV2;

contract Trade{
    struct pri_profile{  //provider profile
		bytes32[] adr_data_hash;
		bytes32[] adr_sample_hash;
		string adr_feature;
		string[] adr_data_plain;
		string[] adr_sample_plain;
		bytes32 Key_data;
		uint V_deposit;
		bool isRegisted;
		bool sampled;
		bool setfeature;
	}

	mapping(address=>pri_profile) public P_profile;
	address public P_col;
	address public P_tc;
	address public P_opt;
	uint public state; //0:pending;   1:reviewing;  2:reviewed  3:closed
	uint public CF_reward;
	uint public CF_deposit;
	uint public CF_percentage;
	uint public CF_expiry;
	uint public start_time;
	string adr_model;
	
	event printstring(string aa);
	event printbytes(bytes ss);
	event printbytes32(bytes32 a);
	event printlength(uint a);
	
	constructor() public{	
    }

    function CreateContract(string memory _modeladr, uint _deposit, uint _percentage) payable public{
    	P_col=msg.sender;
        adr_model=_modeladr; 
		CF_deposit=_deposit;
		CF_percentage=_percentage;
		CF_expiry=2*24;
    	CF_reward=msg.value;
    	start_time=now;
    	state=0; //pending
    }
    
    function ConvertString2Arr(string memory ss, uint one_adr_len) public pure returns(string[] memory){
        //uint one_adr_len=46;
        //the length of one ipfs hash address. eg., QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco
        bytes memory _ss=bytes(ss);
        uint len=_ss.length/one_adr_len;
        string[] memory arr=new string[](len);
        uint k=0;
        for(uint i=0;i<len;i++)
        {
          bytes memory item=new bytes(one_adr_len);
          for(uint j=0; j<one_adr_len; j++)
          {
            item[j]=_ss[k++];
          }
          //emit printbytes(item);
          arr[i]=string(item);
        }
        return arr;
    }
    
    function StrCmp(string memory s1, string memory s2)public pure returns(bool){
        bytes memory _s1=bytes(s1);
        bytes memory _s2=bytes(s2);
        uint len=_s1.length;
        for(uint i=0;i<len;i++){
            if(_s1[i]!=_s2[i])
                return false;
        }
        return true;
    }

    	
    function Registration(bytes32[] memory adr_data_hash) payable public{
		assert(msg.value>=CF_deposit);
		assert(state==0);
		address pid=msg.sender;
		
		assert(P_profile[pid].isRegisted==false);
		
		P_profile[pid].adr_data_hash=adr_data_hash;
		P_profile[pid].V_deposit=msg.value;
		P_profile[pid].isRegisted=true;
	}
	
	
    function SampleData(address pid, bytes32[] memory sample_hash) public{
		assert(msg.sender==P_col && P_profile[pid].isRegisted);
		assert(state==0 && P_profile[pid].sampled==false);
		assert(sample_hash.length*100<=P_profile[pid].adr_data_hash.length*CF_percentage);
		
		P_profile[pid].adr_sample_hash=sample_hash;
		P_profile[pid].sampled=true;
	}
	
	function SubmitFeatures(string memory features) public{
		address pid=msg.sender;
		assert(P_profile[pid].isRegisted);
		assert(state==0 && P_profile[pid].setfeature==false);
		P_profile[pid].adr_feature=features;
		P_profile[pid].setfeature=true;
	}
	
	function SetCandidate(address pid) public{
		assert(msg.sender==P_col && state==0);
		assert(P_profile[pid].isRegisted);
		P_tc=pid;
		state=1; //reviewing
	}

	function SubmitReview(address pid, string memory plain_sample, bytes32 K_data) public{
		assert(msg.sender==pid && pid==P_tc);
		assert(state==1);
		string[] memory adr_sample_plain_arr=ConvertString2Arr(plain_sample, 46);
		uint len=adr_sample_plain_arr.length;
		for(uint i=0;i<len;i++)
		{
		    //check the hash value of each plain address
		    string memory plain_sample_item=adr_sample_plain_arr[i];  //befor hash, one plane addr
		    emit printstring(plain_sample_item);
		    bytes32 hash_item=sha256(bytes(plain_sample_item));
		    
		    emit printbytes32(hash_item);
		    emit printbytes32(P_profile[pid].adr_sample_hash[i]);
		    
		    assert(hash_item==P_profile[pid].adr_sample_hash[i]);
		}
		P_profile[pid].Key_data=K_data;
		P_profile[pid].adr_sample_plain=adr_sample_plain_arr;
	}
    
    function SetOptimal(address pid)public{
        assert(msg.sender==P_col && state==1);
        assert(pid==P_tc && P_opt==address(0));
        P_opt=pid;
        state=2;  //reviewed
    }
    
    function ClaimReward(string memory data_plain)payable public{
        assert(msg.sender==P_opt && state==2);
        string[] memory adr_data_plain_arr=ConvertString2Arr(data_plain, 46);
        uint len=adr_data_plain_arr.length;
        for(uint i=0; i<len; i++)
        {
            string memory plain_data_item=adr_data_plain_arr[i];
            emit printstring(plain_data_item);
            bytes32 hash_item=sha256(bytes(plain_data_item));
            
            emit printbytes32(hash_item);
            emit printbytes32(P_profile[msg.sender].adr_data_hash[i]);
            
            assert(hash_item==P_profile[msg.sender].adr_data_hash[i]);
        }
    	P_profile[msg.sender].adr_data_plain=adr_data_plain_arr;
    	(msg.sender).transfer(CF_reward);
    	state=3; //closed
    }
    
    function ClaimDeposit()payable public{
        assert(state==3 || now>start_time+CF_expiry);
        address pid=msg.sender;
        assert(P_profile[pid].isRegisted);
        (msg.sender).transfer(P_profile[pid].V_deposit);
        P_profile[pid].isRegisted=false;
    }    
}
